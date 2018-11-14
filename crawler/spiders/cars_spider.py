import scrapy

from crawler.items import CarItem


class CarsSpider(scrapy.Spider):
    name = "cars"

    start_urls = [
        "https://www.pexels.com/search/car/",
    ]

    def parse(self, response):
        for car_item in response.css(".photo-item"):
            url = car_item.css('img::attr("src")').extract_first().split("?")[0]
            yield CarItem(file_urls=[url])

        next_page = response.css('.next_page::attr("href")').extract_first()
        if next_page is not None:
            yield response.follow(next_page, self.parse)
