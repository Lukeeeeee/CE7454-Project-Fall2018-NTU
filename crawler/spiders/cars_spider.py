import time

import scrapy
from selenium import webdriver

from crawler.items import CarItem


class CarsSpider(scrapy.Spider):
    name = "cars"

    def __init__(self):
        self.driver = webdriver.Firefox()

    def start_requests(self):
        main_url = "https://www.pexels.com/search/car/"
        yield scrapy.Request(url=main_url, callback=self.parse)

    def parse(self, response):
        self.driver.get(response.url)
        count = 0
        while count < 100:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            print(count)
            count += 1
        car_urls = self.driver.find_elements_by_xpath('//article/a[2]')
        car_id = 1
        for text in car_urls:
            url = text.get_attribute("href")
            url = url.split('?')[0]
            print(url)
            yield CarItem(file_urls=[url])
            car_id += 1
            time.sleep(0.25)
