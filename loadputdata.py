import csv
from datetime import datetime

path = '/Users/hansbecker1/Studie/6. EUR 2019-2020/4 Blok 4/putdata.csv'

file = open(path, 'r')
reader = csv.reader(file)
next(reader)

putdata = list()

for line in reader:

    #Check if traded recently
    if str(line[2]) != '':
        continue

    #check if implied volatility is present
    if str(line[7]) == '':
        continue

    s = line[0]
    s = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
    date = s

    s = line[1]
    s = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
    expdate = s

    strike_price = float(line[4])
    best_bid = float(line[5])
    best_offer = float(line[6])
    impl_volatility = float(line[7])
    time_to_maturity = expdate - date
    moneyness = ((best_bid + best_offer)/2) / strike_price
    t = [date, expdate, strike_price, best_bid, best_offer, impl_volatility, time_to_maturity, moneyness]
    putdata.append(t)

print(len(putdata))
print(putdata[1][1], putdata[1][4])

#with open('/Users/hansbecker1/Studie/6. EUR 2019-2020/4 Blok 4/testexport.csv', 'wb') as myfile

