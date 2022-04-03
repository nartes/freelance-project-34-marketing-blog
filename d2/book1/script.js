$(window).on('load', () => {
  window.context = {};
  window.context.books = [];

  var synth = window.speechSynthesis;

  var inputForm = document.querySelector('form');
  var inputTxt = document.querySelector('.txt');
  var voiceSelect = document.querySelector('select');

  var pitch = document.querySelector('#pitch');
  var pitchValue = document.querySelector('.pitch-value');
  var rate = document.querySelector('#rate');
  var rateValue = document.querySelector('.rate-value');

  var voices = [];


  context.nosleep_timer = null;

  context.ui = {
    voice_settings_div: $('.voice-settings'),
    voice_select: $('.voice-select'),
    status_pre: $('.status'),
    books_select: $('.screen .widget select[name=book]'),
    current_sentence_input:
      $('.screen .widget input[name=current-sentence]'),
    total_sentences_input:
      $('.screen .widget input[name=total-sentences]'),
    read_aloud:
      $('.screen .widget input[name=read-aloud]'),
    debug:
      $('.screen .widget input[name=debug]'),
  };
  context.sentences = null;
  context.pending_stop = false;
  context.current_book = null;
  context.nosleep = new NoSleep();
  context.is_debug = false;
  context.log = {
    error: [],
    info: [],
  };
  context.callbacks = {
    log_error: (msg) => {
      if (context.is_debug)
      {
        console.error(msg);
        context.log.error.push(msg);
      }
    },
    enable_no_sleep: () => {
      if (context.nosleep_timer != null)
      {
        context.callbacks.log_error('running already');
      }

      context.nosleep_timer = setInterval(
        () => {
          location.hash = 'nosleep' + Math.random();
          context.callbacks.update_status();
          /*
          if ('vibrate' in window.navigator)
          {
            window.navigator.vibrate(200);
          }
          */
        }, 1000
      );
    },
    get_state: () => {
      let t1 = localStorage['state'];
      if (t1)
      {
        return JSON.parse(t1);
      }
      else
      {
        return {};
      }
    },
    get_cookie: (key) => {
      /*
      return document.cookie.split('; ').map(
        (o) => o.split('=')
      ).reduce(
        (b, a) => {
          if (a.length == 2) {b[a[0]] = a[1]};
          return b
        },
        {}
      )[key];
      */
      let t1 = localStorage['state'];
      if (t1 != undefined)
      {
        let t2 = JSON.parse(t1);
        return t2[key];
      }
      else
      {
        return undefined;
      }
    },
    set_cookie: (key, value) => {
      let state = context.callbacks.get_state('state');

      state[key] = value;

      //document.cookie = `${key}=${value};`;
      localStorage['state'] = JSON.stringify(state);

      context.callbacks.update_status();
    },
    disable_no_sleep: () => {
      if (context.nosleep_timer == null)
      {
        context.callbacks.log_error('nothing is running');
      }
      clearInterval(context.nosleep_timer);
      location.hash = '';
      context.nosleep_timer = null;
      synth.cancel();
    },
    continuous_reading: async() => {
      if (context.is_reading)
      {
        context.pending_stop = true;
        return;
      }
      context.is_reading = true;
      context.nosleep.enable();
      context.callbacks.enable_no_sleep();
      context.ui.voice_settings_div.addClass('hidden');
      context.ui.current_sentence_input.attr(
        'disabled',
        'disabled'
      );

      while (
        context.callbacks.get_cookie('sentence_id') < context.sentences.length &&
        !context.pending_stop 
      )
      {
        let sentence =
          context.sentences[context.callbacks.get_cookie('sentence_id')];
        //context.callbacks.log_error('start');
        try {
          await context.read_aloud(
            context.sentences[
              context.callbacks.get_cookie('sentence_id')
            ]
          );
        } catch (e) {
          context.callbacks.log_error(e);
        }
        //context.callbacks.log_error('finished');
        if (!context.pending_stop)
        {
          context.callbacks.set_cookie(
            'sentence_id',
            context.callbacks.get_cookie('sentence_id') + 1
          );
        }
      }
      context.pending_stop = false;
      context.ui.current_sentence_input.removeAttr('disabled');
      context.nosleep.disable();
      context.ui.voice_settings_div.removeClass('hidden');
      context.callbacks.disable_no_sleep();
      context.is_reading = false;
    },
    update_status: () => {
      let data = {};
      data.state = context.callbacks.get_state();
      if (
        context.callbacks.get_cookie('sentence_id') != null &&
        context.sentences != null &&
        context.callbacks.get_cookie('sentence_id') < context.sentences.length
      )
      {
        data.sentence = context.sentences[context.callbacks.get_cookie('sentence_id')];
      }
      data.pending_stop = context.pending_stop;
      data.is_reading = context.is_reading;
      data.log = context.log;
      context.ui.current_sentence_input.val(
        context.callbacks.get_cookie('sentence_id')
      );
      data.timestamp = (new Date());
      data.version = 'v0.1.7';
      data.speech_synthesis = {
        paused: synth.paused,
        pending: synth.pending,
        speaking: synth.speaking,
      };
      /*
      if (!synth.speaking && context.is_reading)
      {
        synth.cancel();
      }
      */
      context.ui.status_pre.text(
        JSON.stringify(
          data,
          null,
          4,
        )
      );
    },
    ui_read_aloud_on_click: async() => {
      let book_id = parseInt(context.ui.books_select.val());
      if (context.current_book != book_id)
      {
        context.current_book = book_id;
        context.sentences =
          context.books[
            context.current_book
          ].replaceAll(/([\.\?\!])\s+/g,'$1\n')
          .split('\n');
        context.ui.total_sentences_input.val(
          context.sentences.length,
        );
        {
          let state = context.callbacks.get_state();
        }
      }
      if (
        context.ui.current_sentence_input.val() != ''
      )
      {
        try{
          let sentence_id = parseInt(
            context.ui.current_sentence_input.val()
          );

          if (
            sentence_id >= 0 &&
            sentence_id < context.sentences.length
          )
          {
            context.callbacks.set_cookie(
              'sentence_id',
              sentence_id
            );
          }
        } catch (e) {
          context.callbacks.log_error(e);
        }
      }
      if (context.is_reading && !context.pending_stop)
      {
        context.pending_stop = true;
      }
      else
      {
        context.callbacks.continuous_reading();
      }
    },
    populateVoiceList: () => {
      voices = synth.getVoices().sort(function (a, b) {
          const aname = a.name.toUpperCase(), bname = b.name.toUpperCase();
          if ( aname < bname ) return -1;
          else if ( aname == bname ) return 0;
          else return +1;
      });
      //var selectedIndex = voiceSelect.selectedIndex < 0 ? 0 : voiceSelect.selectedIndex;
      voiceSelect.innerHTML = '';
      for(i = 0; i < voices.length ; i++) {
        var option = document.createElement('option');
        option.textContent = voices[i].name + ' (' + voices[i].lang + ')';
        
        if(voices[i].default) {
          option.textContent += ' -- DEFAULT';
        }


        {
          let voice = context.callbacks.get_cookie('voice');
          if (voice && option.textContent == voice)
          {
            $(option).attr('selected', 'selected');
          }
        }

        option.setAttribute('data-lang', voices[i].lang);
        option.setAttribute('data-name', voices[i].name);
        voiceSelect.appendChild(option);
      }

      //voiceSelect.selectedIndex = selectedIndex;
    },
    init: () => {
      let state = context.callbacks.get_state();
      context.ui.voice_select.val(state.voice);
      if (!state.book_id)
      {
        context.callbacks.set_cookie(
          'book_id',
          0,
        );
      }
      if (!state.sentence_id)
      {
        context.callbacks.set_cookie(
          'sentence_id',
          0,
        );
      }
      if (state.book_id)
      {
        context.ui.books_select.find(
          '>option',
        ).eq(state.book_id).attr('selected', 'selected');
      }
      if (state.sentence_id)
      {
        context.ui.current_sentence_input.val(
          state.sentence_id,
        );
      }
    },
  };
  context.callbacks.populateVoiceList();
  if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.onvoiceschanged = context.callbacks.populateVoiceList;
  }

  context.callbacks.init();

  context.ui.read_aloud.on(
    'click',
    context.callbacks.ui_read_aloud_on_click,
  );
  context.ui.voice_select.on(
    'change',
    () => {
      context.callbacks.set_cookie(
        'voice',
        context.ui.voice_select.val()
      );
    }
  );
  context.ui.debug.on(
    'click',
    () => {
      if (context.is_debug)
      {
        context.is_debug = false;
      }
      else
      {
        context.is_debug = true;
      }
      context.callbacks.update_status();
    }
  );
  context.read_aloud = async (raw_line) => {
    line = raw_line.trim();
    if (line.length == 0)
    {
      return;
    }
    let sleep_detect = null;
    let exit = () => {
      if (sleep_detect != null)
      {
        clearInterval(sleep_detect);
      }
    }
    return new Promise((response, reject) => {
      if (synth.speaking) {
          context.callbacks.log_error('speechSynthesis.speaking');
          if (reject != undefined)
          {
            reject('error');
          }
          return;
      }
      let utterThis = new SpeechSynthesisUtterance(line);
      utterThis.onend = function (event) {
          exit();
          context.callbacks.log_error(
            'SpeechSynthesisUtterance.onend ' + event.error
          );
          if (response != undefined)
          {
            response('done ' + event.error);
          }
      }
      utterThis.onpause = function (event) {
          exit();
          context.callbacks.log_error('SpeechSynthesisUtterance.onpause');
          if (reject != undefined)
          {
            reject('paused ' + event.error);
          }
      }
      utterThis.onerror = function (event) {
          exit();
          context.callbacks.log_error(
            'SpeechSynthesisUtterance.onerror ' + event.error
          );
          if (reject != undefined)
          {
            reject('error ' + event.error);
          }
      }
      let selectedOption = voiceSelect.selectedOptions[0].getAttribute('data-name');
      for(i = 0; i < voices.length ; i++) {
        if(voices[i].name === selectedOption) {
          utterThis.voice = voices[i];
          break;
        }
      }
      //window.alert('fuck3');
      utterThis.pitch = pitch.value;
      utterThis.rate = rate.value;
      synth.speak(utterThis);
      let silence_count = 0;
      sleep_detect = setInterval(
        () => {
          if (!synth.speaking)
          {
            context.callbacks.log_error(
              'silence count is ' + silence_count
            )

            ++silence_count;
          }

          if (silence_count == 3 || context.pending_stop)
          {
            exit();
            if (context.pending_stop)
            {
              synth.cancel();
              reject('pending stop');
            }
            else
            {
              context.callbacks.log_error('phone is sleeping, retry');
              response('utterance is not present');
            }
            /*
            context.read_aloud(
              line
            ).then(response).catch(reject);
            */
          }
        },
        100,
      );
    });
  }

  function speak(){
    let line = inputTxt.value;
    if (line !== '') {
      context.read_aloud(line);
    }
  }

  inputForm.onsubmit = function(event) {
    event.preventDefault();

    speak();

    inputTxt.blur();
  }

  pitch.onchange = function() {
    pitchValue.textContent = pitch.value;
  }

  rate.onchange = function() {
    rateValue.textContent = rate.value;
  }

  voiceSelect.onchange = function(){
    speak();
  }
});
