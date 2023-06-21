import datetime

times = '''
0.12814516170024873	0.05446938016414642
1.3223235380649567	1.8037164335250855
35.579042961597445	152.0151873922348
'''.strip().replace('\t', ',').replace('\n', ',').split(',')
times = [float(x) for x in times]

seconds = 10000 * sum([times[0], times[1]]) + 1000 * sum([times[2], times[3]]) + 100 * sum([times[4], times[5]])
print(str(datetime.timedelta(seconds=seconds)))