(() => {
    window.context.c1 = {
        m1: (target, key_map, timeout, freq, mouse_amplifier) => {
            if (mouse_amplifier == undefined)
            {
                mouse_amplifier = 100;
            }

            if (freq == undefined)
            {
                freq = 50;
            }
            if (timeout == undefined)
            {
                timeout = 40;
            }

            let ctx = {
                timer_id: null,
                game_pad: null
            };
            ctx.game_pad = new window.context.game_pad.GamePads();
            if (target == undefined)
            {
                throw "NOT_IMPLEMENTED_ERROR";
            }
            if (key_map == undefined)
            {
                throw "NOT_IMPLEMENTED_ERROR";
            }
            if (timeout == undefined)
            {
                throw "NOT_IMPLEMENTED_ERROR";
            }
            let t6 = null;
            let t7 = null;
            ctx.timer_id = setInterval(
                () => {
                    ctx.game_pad.update();
                    if (
                        ctx.game_pad.current.length < 1 &&
                        ctx.game_pad.previous.length < 1
                    )
                    {
                        return;
                    }

                    let o3 = {};
                    let o4 = {}
                    let t4 = ctx.game_pad.previous[0];
                    let t5 = ctx.game_pad.current[0];

                    if (t4 == undefined || t5 == undefined)
                    {
                        return;
                    }
                    t6 = t7;
                    t7 = {};

                    t5.pad.buttons.forEach(
                        (o, i) => {
                            o4[i] = t5.map.buttons[i];
                            o4[o4[i]] = i;

                            o3[
                                t5.map.buttons[i]
                            ] = o.value;
                            t7[o4[i]] = o.value;
                        }
                    );
                    for (k in o3)
                    {
                        if (key_map[k] != undefined)
                        {
                            let event_type = null;

                            if (
                                o3[k] == 1 && (t6 == null || t6[k] == 0)
                            )
                            {
                                event_type = "keydown";
                            }
                            else if (
                                o3[k] == 0 && (
                                    t6 != null && t6[k] == 1
                                )
                            )
                            {
                                event_type = "keyup";
                            }

                            if (event_type != null)
                            {
                                console.log([target, key_map[k], event_type]);
                                window.context.c1.m2(
                                    target,
                                    key_map[k],
                                    event_type,
                                );
                            }
                        }
                    }
                    {
                        let dx = t5.pad.axes[1];
                        let dy = t5.pad.axes[2];

                        if (Math.abs(dx) > 0.05 || Math.abs(dy) > 0.05)
                        {
                            window.context.drag_mouse(
                                dx * mouse_amplifier,
                                dy * mouse_amplifier,
                            );
                        }
                    }
                    if (o3.PAD_L_SHOULDER_1 || o3.PAD_R_SHOULDER_1)
                    {
                        window.context.drag_mouse(
                            0,
                            0,
                            [
                                0,
                            ]
                        );
                    }
                    if (o3.PAD_SELECT == 1)
                    {
                        clearInterval(ctx.timer_id);
                        console.log(
                            `
                                d2.gamepad.script window.context.c1.m1
                                clearInterval ${ctx.timer_id}
                            `
                        );
                    }
                },
                freq,
            );
            return ctx;
        },
        m4: (key_code) => {
            let t1 = null;
            for (let k in window.context.game_pad.KEYS)
            {
                if (window.context.game_pad.KEYS[k] == key_code)
                {
                    t1 = k;
                }
            }
            if (t1 == null)
            {
                throw `NOT_IMPLEMENTED_ERROR`;
            }
            return {
                code: key_code,
                lable: t1,
            };
        },
        m3: (target, key_code, timeout) => {
            window.context.c1.m2(target, key_code, `keydown`);

            if (!(timeout > 0 && timeout < 2000))
            {
                throw `NOT_IMPLEMENTED_ERROR`;
            }

            return setTimeout(
                () => {
                    window.context.c1.m2(target, key_code, `keyup`);
                },
                timeout,
            );
        },
        m5: (a, b) => {
            return b.findIndex((o) => o == a);
        },
        m2: (target, key_code, key_event_name) => {
            let t1 = window.context.c1.m4(key_code);

            if (
                window.context.c1.m5(
                    key_event_name,
                    [`keydown`, `keyup`]
                ) == -1
            )
            {
                throw `NOT_IMPLEMENTED_ERROR`;
            }

            target.dispatchEvent(
                new KeyboardEvent(
                    key_event_name,
                    {
                        key: t1.label,
                        keyCode: t1.code, // example values.
                        code: t1.code, // put everything you need in this object.
                        which: t1.code,
                        shiftKey: false, // you don't need to include values
                        ctrlKey: false,  // if you aren't going to use them.
                        metaKey: false   // these are here for example's sake.
                    }
                )
            );
        },
    };
})();
