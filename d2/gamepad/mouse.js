if (window.context == undefined)
{
    window.context = {};
}

window.context.drag_mouse = (dx, dy, buttons, element) => {
    let sx = null;
    let sy = null;

    if (element == undefined)
    {
        element = document.body;
    }
    if (buttons == undefined)
    {
        buttons = []
    }

    {
        let t1 = element.getBoundingClientRect();
        sx = t1.left + t1.width / 2;
        sy = t1.top + t1.height / 2;
    }
    //console.log([sx, sy]);

    t2 = (btn, mvx, mvy, clx, cly, t) => {
        return new MouseEvent(
            t,
            {
                type: t,
                button: btn,
                target: element,
                movementX: mvx,
                movementY: mvy,
                clientX: clx,
                clientY: cly,
                shiftKey: false, // you don't need to include values
                ctrlKey: false,  // if you aren't going to use them.
                metaKey: false   // these are here for example's sake.
            }
        );
    }

    t3 = [
        t2(-1, 0, 0, sx, sy, "mousemove"),
        t2(pc.MOUSEBUTTON_LEFT, 0, 0, sx, sy, "mousedown"),
        t2(-1, 0, 0, sx, sy, "mousemove"),
        t2(-1, dx, dy, sx + dx, sy + dy, "mousemove"),
        t2(-1, 0, 0, sx + dx, sy + dy, "mousemove"),
        t2(-1, 0, 0, sx + dx, sy + dy, "mousemove"),
        t2(pc.MOUSEBUTTON_LEFT, 0, 0, sx + dx, sy + dy, "mouseup"),
    ];
    if (true)
    {
        window.dispatchEvent(t3[0]);
    }
    if (true)
    {
        for (let button in buttons)
        {
            let event = t2(
                button, 0, 0, sx, sy, "mousedown"
            );
            window.dispatchEvent(event);
        }
    }
    t3.slice(3, 4).forEach(
        (o) => window.dispatchEvent(o)
    );
    //t3.slice(4, 5).forEach(
    //    (o) => window.dispatchEvent(o)
    //);
    //t3.slice(5, 6).forEach(
    //    (o) => window.dispatchEvent(o)
    //);

    if (false)
    {
        for (let button in buttons)
        {
            let event = t2(
                button, 0, 0, sx + dx, sy + dy, "mouseup",
            );
            window.dispatchEvent(event);
        }
    }

    window.dispatchEvent(t3[6]);
}
