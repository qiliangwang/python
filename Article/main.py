from scrapy.cmdline import execute
import sys
import os
main_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(main_dir)
execute(['scrapy', 'crawl', 'job'])
