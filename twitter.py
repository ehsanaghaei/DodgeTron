import pandas as pd
import snscrape.modules.twitter as snw
hashtags = ["#TempleMade"]
tlist = list()
for i, tweet in enumerate(snw.TwitterSearchScraper(hashtags[-1] + 'since:2023-01-01').get_items()):
  print(tweet)