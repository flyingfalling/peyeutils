
for f in `ls *.png`; do
    #magick convert $f $f".pdf"
    echo "DOING " $f
    magick $f $f".pdf"
done

pdfunite *.pdf result.pdf
