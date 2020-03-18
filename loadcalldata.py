import csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt



path = '/Users/hansbecker1/Studie/6. EUR 2019-2020/4 Blok 4/calldata.csv'

file = open(path, 'r')
reader = csv.reader(file)
next(reader)

calldata = list()

for line in reader:

    #Check if traded recently
    if str(line[2]) != '':
        continue

    #check if implied volatility is present
    if str(line[7]) == '':
        continue

    #exclude options with a strike of zero
    if float(line[4]) == 0:
        continue

    s = line[0]
    s = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
    date = s

    s = line[1]
    s = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
    expdate = s

    strike_price = float(line[4])/100
    best_bid = float(line[5])
    best_offer = float(line[6])
    impl_volatility = float(line[7])
    time_to_maturity = expdate - date
    time_to_maturity = time_to_maturity.days
    t = [date, expdate, strike_price, best_bid, best_offer, impl_volatility, time_to_maturity]
    calldata.append(t)

print(len(calldata))
#x = [i[2]/1000 for i in calldata]
#y = [i[5] for i in calldata]
#plt.scatter(x,y, s=1)
#plt.xlabel('Strike')
#plt.ylabel('Implied vol')
#plt.show()

#x = [i[6] for i in calldata]
#plt.hist(x, bins=200)
#plt.show()

x = [i[4] for i in calldata]
plt.hist(x, bins=200)
plt.show()

#path = '/Users/hansbecker1/Studie/6. EUR 2019-2020/4 Blok 4/spxlevel.csv'
