(() => {
    window.context.runtime = {};

    window.context.drag_mouse(100, 0, [pc.MOUSEBUTTON_LEFT], $$('canvas')[0])
    window.context.runtime.m1_ctx = window.context.c1.m1(
        window,
        {
            "PAD_FACE_1": window.context.game_pad.KEYS.KEY_W,
            "PAD_FACE_2": window.context.game_pad.KEYS.KEY_D,
            "PAD_FACE_3": window.context.game_pad.KEYS.KEY_S,
            "PAD_FACE_4": window.context.game_pad.KEYS.KEY_A,
        },
    );
})();
