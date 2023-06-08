(
    echo -e "#!/bin/sh\necho \$CPANEL_SSH_PASS" > /tmp/askpass.sh &&
    chmod +x /tmp/askpass.sh &&
    DISPLAY=blah \
    SSH_ASKPASS=/tmp/askpass.sh \
    SSH_ASKPASS_REQUIRE=prefer \
    ssh-add $T1
) &&
ssh-add -l &&
while true; do
    for k in $(seq 9051 15000);
    do
        date; echo $k;
        ssh \
            -C -o 'ServerAliveInterval=2' -o 'ConnectTimeout=2' \
	    -o 'ExitOnForwardFailure=yes' \
            -i $T1 \
            $T3@$T2  \
            'exec tmp/busybox pkill -f proxy.json';
        ssh \
            -C -o 'ServerAliveInterval=2' -o 'ConnectTimeout=2' \
	    -o 'ExitOnForwardFailure=yes' \
            -i $T1 \
            $T3@$T2  \
            'exec tmp/busybox pkill -f 99999942';
        ssh \
            -C -o 'ServerAliveInterval=2' -o 'ConnectTimeout=2' \
	    -o 'ExitOnForwardFailure=yes' \
            -i $T1 \
            $T3@$T2  \
            -R 127.0.0.1:$k:app:80 \
            '(echo [\"127.0.0.1:'$k'\"] > /home/'$T3'/proxy.json && exec sleep 99999942)';
        sleep 1;
    done;
done;
