#! /bin/sh

filename='quote_files_names'
filelines=`cat $filename`

for line in $filelines ; do
   date=${line: -8};
   sed "s/.$/,$date/" $line | sed 's/  */-/g' | sed 's/-/,/' > tmp
   sed "s/-,$date$/,$date/" tmp | sed '1 s/^.*$/A,B,Date/' > $line
done

rm tmp
