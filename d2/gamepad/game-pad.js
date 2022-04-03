(() => {
    const KEYS = {
        "KEY_0":48,"KEY_1":49,"KEY_2":50,"KEY_3":51,"KEY_4":52,"KEY_5":53,"KEY_6":54,"KEY_7":55,"KEY_8":56,"KEY_9":57,"KEY_A":65,"KEY_ADD":107,"KEY_ALT":18,"KEY_B":66,"KEY_BACKSPACE":8,"KEY_BACK_SLASH":220,"KEY_C":67,"KEY_CAPS_LOCK":20,"KEY_CLOSE_BRACKET":221,"KEY_COMMA":188,"KEY_CONTEXT_MENU":93,"KEY_CONTROL":17,"KEY_D":68,"KEY_DECIMAL":110,"KEY_DELETE":46,"KEY_DIVIDE":111,"KEY_DOWN":40,"KEY_E":69,"KEY_END":35,"KEY_ENTER":13,"KEY_EQUAL":61,"KEY_ESCAPE":27,"KEY_F":70,"KEY_F1":112,"KEY_F10":121,"KEY_F11":122,"KEY_F12":123,"KEY_F2":113,"KEY_F3":114,"KEY_F4":115,"KEY_F5":116,"KEY_F6":117,"KEY_F7":118,"KEY_F8":119,"KEY_F9":120,"KEY_G":71,"KEY_H":72,"KEY_HOME":36,"KEY_I":73,"KEY_INSERT":45,"KEY_J":74,"KEY_K":75,"KEY_L":76,"KEY_LEFT":37,"KEY_M":77,"KEY_META":224,"KEY_MULTIPLY":106,"KEY_N":78,"KEY_NUMPAD_0":96,"KEY_NUMPAD_1":97,"KEY_NUMPAD_2":98,"KEY_NUMPAD_3":99,"KEY_NUMPAD_4":100,"KEY_NUMPAD_5":101,"KEY_NUMPAD_6":102,"KEY_NUMPAD_7":103,"KEY_NUMPAD_8":104,"KEY_NUMPAD_9":105,"KEY_O":79,"KEY_OPEN_BRACKET":219,"KEY_P":80,"KEY_PAGE_DOWN":34,"KEY_PAGE_UP":33,"KEY_PAUSE":19,"KEY_PERIOD":190,"KEY_PRINT_SCREEN":44,"KEY_Q":81,"KEY_R":82,"KEY_RETURN":13,"KEY_RIGHT":39,"KEY_S":83,"KEY_SEMICOLON":59,"KEY_SEPARATOR":108,"KEY_SHIFT":16,"KEY_SLASH":191,"KEY_SPACE":32,"KEY_SUBTRACT":109,"KEY_T":84,"KEY_TAB":9,"KEY_U":85,"KEY_UP":38,"KEY_V":86,"KEY_W":87,"KEY_WINDOWS":91,"KEY_X":88,"KEY_Y":89,"KEY_Z":90
    };
    const MAPS = {
        DEFAULT: {
        buttons: [
            // Face buttons
            'PAD_FACE_1',
            'PAD_FACE_2',
            'PAD_FACE_3',
            'PAD_FACE_4',

            // Shoulder buttons
            'PAD_L_SHOULDER_1',
            'PAD_R_SHOULDER_1',
            'PAD_L_SHOULDER_2',
            'PAD_R_SHOULDER_2',

            // Other buttons
            'PAD_SELECT',
            'PAD_START',
            'PAD_L_STICK_BUTTON',
            'PAD_R_STICK_BUTTON',

            // D Pad
            'PAD_UP',
            'PAD_DOWN',
            'PAD_LEFT',
            'PAD_RIGHT',

             // Vendor specific button
            'PAD_VENDOR'
        ],

        axes: [
            // Analogue Sticks
            'PAD_L_STICK_X',
            'PAD_L_STICK_Y',
            'PAD_R_STICK_X',
            'PAD_R_STICK_Y'
        ]
        },

        PS3: {
        buttons: [
            // X, O, TRI, SQ
            'PAD_FACE_1',
            'PAD_FACE_2',
            'PAD_FACE_4',
            'PAD_FACE_3',

            // Shoulder buttons
            'PAD_L_SHOULDER_1',
            'PAD_R_SHOULDER_1',
            'PAD_L_SHOULDER_2',
            'PAD_R_SHOULDER_2',

            // Other buttons
            'PAD_SELECT',
            'PAD_START',
            'PAD_L_STICK_BUTTON',
            'PAD_R_STICK_BUTTON',

            // D Pad
            'PAD_UP',
            'PAD_DOWN',
            'PAD_LEFT',
            'PAD_RIGHT',

            'PAD_VENDOR'
        ],

        axes: [
            // Analogue Sticks
            'PAD_L_STICK_X',
            'PAD_L_STICK_Y',
            'PAD_R_STICK_X',
            'PAD_R_STICK_Y'
        ]
        }
    };

    const PRODUCT_CODES = {
        'Product: 0268': 'PS3'
    };

    /**
     * Input handler for accessing GamePad input.
     */
    class GamePads {
        /**
         * Create a new GamePads instance.
         */
        constructor() {
            this.gamepadsSupported = !!navigator.getGamepads || !!navigator.webkitGetGamepads;

            this.current = [];
            this.previous = [];

            this.deadZone = 0.25;
        }

        /**
         * Update the current and previous state of the gamepads. This must be called every frame for
         * `wasPressed` to work.
         */
        update() {
            // move current buttons status into previous array
            for (let i = 0, l = this.current.length; i < l; i++) {
                const buttons = this.current[i].pad.buttons;
                const buttonsLen = buttons.length;
                for (let j = 0; j < buttonsLen; j++) {
                if (this.previous[i] === undefined) {
                    this.previous[i] = [];
                }
                this.previous[i][j] = buttons[j].pressed;
                }
            }

            // update current
            this.poll(this.current);
        }

        /**
         * Poll for the latest data from the gamepad API.
         *
         * @param {object[]} [pads] - An optional array used to receive the gamepads mapping. This
         * array will be returned by this function.
         * @returns {object[]} An array of gamepads and mappings for the model of gamepad that is
         * attached.
         * @example
         * var gamepads = new pc.GamePads();
         * var pads = gamepads.poll();
         */
        poll(pads = []) {
            if (pads.length > 0) {
                pads.length = 0;
            }

            if (this.gamepadsSupported) {
                const padDevices = navigator.getGamepads ? navigator.getGamepads() : navigator.webkitGetGamepads();
                for (let i = 0, len = padDevices.length; i < len; i++) {
                if (padDevices[i]) {
                    pads.push({
                    map: this.getMap(padDevices[i]),
                    pad: padDevices[i]
                    });
                }
                }
            }
            return pads;
        }

        getMap(pad) {
            for (const code in PRODUCT_CODES) {
                if (pad.id.indexOf(code) >= 0) {
                return MAPS[PRODUCT_CODES[code]];
                }
            }

            return MAPS.DEFAULT;
        }

        /**
         * Returns true if the button on the pad requested is pressed.
         *
         * @param {number} index - The index of the pad to check, use constants {@link PAD_1},
         * {@link PAD_2}, etc.
         * @param {number} button - The button to test, use constants {@link PAD_FACE_1}, etc.
         * @returns {boolean} True if the button is pressed.
         */
        isPressed(index, button) {
            if (!this.current[index]) {
                return false;
            }

            const key = this.current[index].map.buttons[button];
            return this.current[index].pad.buttons[pc[key]].pressed;
        }

        /**
         * Returns true if the button was pressed since the last frame.
         *
         * @param {number} index - The index of the pad to check, use constants {@link PAD_1},
         * {@link PAD_2}, etc.
         * @param {number} button - The button to test, use constants {@link PAD_FACE_1}, etc.
         * @returns {boolean} True if the button was pressed since the last frame.
         */
        wasPressed(index, button) {
            if (!this.current[index]) {
                return false;
            }

            const key = this.current[index].map.buttons[button];
            const i = pc[key];

            // Previous pad buttons may not have been populated yet
            // If this is the first time frame a pad has been detected
            return this.current[index].pad.buttons[i].pressed && !(this.previous[index] && this.previous[index][i]);
        }

        /**
         * Returns true if the button was released since the last frame.
         *
         * @param {number} index - The index of the pad to check, use constants {@link PAD_1},
         * {@link PAD_2}, etc.
         * @param {number} button - The button to test, use constants {@link PAD_FACE_1}, etc.
         * @returns {boolean} True if the button was released since the last frame.
         */
        wasReleased(index, button) {
            if (!this.current[index]) {
                return false;
            }

            const key = this.current[index].map.buttons[button];
            const i = pc[key];

            // Previous pad buttons may not have been populated yet
            // If this is the first time frame a pad has been detected
            return !this.current[index].pad.buttons[i].pressed && (this.previous[index] && this.previous[index][i]);
        }

        /**
         * Get the value of one of the analogue axes of the pad.
         *
         * @param {number} index - The index of the pad to check, use constants {@link PAD_1},
         * {@link PAD_2}, etc.
         * @param {number} axes - The axes to get the value of, use constants {@link PAD_L_STICK_X},
         * etc.
         * @returns {number} The value of the axis between -1 and 1.
         */
        getAxis(index, axes) {
            if (!this.current[index]) {
                return 0;
            }

            const key = this.current[index].map.axes[axes];
            let value = this.current[index].pad.axes[pc[key]];

            if (Math.abs(value) < this.deadZone) {
                value = 0;
            }
            return value;
        }
    }

    if (window['context'] == undefined)
    {
        window.context = {};
    }
    window.context.game_pad = {
        GamePads: GamePads,
        MAPS: MAPS,
        KEYS: KEYS,
    };
})();
//export { GamePads };
