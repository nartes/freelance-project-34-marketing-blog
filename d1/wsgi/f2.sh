echo START; pgrep -f -a 'python.*wsgi-loader'; pgrep -f -a 'python.*wsgi-loader' | awk '{print $1}' | xargs kill -s SIGINT; echo AFTER; pgrep -f -a 'python.*wsgi-loader';
