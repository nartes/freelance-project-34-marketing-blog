$@  'mkdir -p $HOME/p1/p1; mkdir -p $HOME/tmp'
cat d1/f2.py | $@ sh -c 'cat > $HOME/p1/p1/passenger_wsgi.py'
cat tmp/wsgi_config.json | $@ sh -c 'cat > $HOME/p1/p1/wsgi_config.json'
cat d1/wsgi/.htaccess | $@ sh -c 'cat > $HOME/public_html/.htaccess'
cat d1/wsgi/busybox | $@ sh -c 'cat > $HOME/tmp/busybox'
cat d1/wsgi/f2.sh | $@ sh -c 'cat > $HOME/p1/p1/f2.sh'
$@ sh p1/p1/f2.sh
