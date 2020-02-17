python -x %0 %*     &goto: eof
import sys
sys.stdout=open("output.txt","w")
import wincnn
wincnn.showCookToomFilter((0,1,-1), 2, 3)
sys.stdout.close()
