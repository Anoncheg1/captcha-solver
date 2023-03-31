#!/bin/sh
r=0
for f in ./wav0/*_*.* ; do
    # if [ $i -eq 40 ] ; then break ; fi
    # curl -X POST -F "file=@$f" http://srv-whisper.rnb.com/audio_captcha
    right=$(echo "$f" | sed "s/.*_\(.*\).wav/\1/")
    ret=$(curl -X POST -F "file=@$f" http://localhost:8080/audio_captcha)
    # ret=$(curl -X POST -F "file=@$f" http://srv-whisper.rnb.com/audio_captcha)
    test=$(echo -e $ret | grep -o \"[^\"]*\" | head -n2 | tail -n1)
    echo $right ${test:1:5} , $ret, $f
    if [ $right = ${test:1:5} ] ; then
	echo same
	r=$((r+1))
    else
	echo not same
    fi
    i=$((i+1))
    python -c "print('accuracy:', $r/$i, 'iteration:', $i)"
done
