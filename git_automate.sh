git add .

date +"%h %d %y %r"

git commit -m $(printf "%s" $(date +"%h %d %y %r"))

git push origin master
