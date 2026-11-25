"""Scrape remedy texts from materiamedica.info into one .txt file per remedy.

    scrapy runspider data_collection/scrape_materia_medica.py \
        -a source=james-tyler-kent -a out_dir=medicines_data [-a limit=20]

`source` is the author slug used in the site's URLs (see sources.txt for the seven used in HFT).
Check the site's terms of use and be polite (the default settings below add a download delay).
"""
import re
from pathlib import Path

import scrapy

BASE = "https://www.materiamedica.info/en/materia-medica/{source}/index"


class MateriaMedicaSpider(scrapy.Spider):
    name = "materia_medica"
    custom_settings = {"DOWNLOAD_DELAY": 1.0, "ROBOTSTXT_OBEY": True}

    def __init__(self, source="james-tyler-kent", out_dir="medicines_data", limit=None, **kwargs):
        super().__init__(**kwargs)
        self.start_urls = [BASE.format(source=source)]
        self.out_dir = Path(out_dir) / source
        self.limit = int(limit) if limit else None
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, response):
        links = response.css("div.remedy_list a")
        for a in links[: self.limit]:
            name = (a.css("::text").get() or "").strip().lower()
            href = a.attrib.get("href")
            if name and href:
                yield response.follow(href, self.parse_remedy, meta={"remedy": name})

    def parse_remedy(self, response):
        header = response.xpath("//h1/following-sibling::text()[normalize-space()]").getall()
        body = response.xpath("//p[not(ancestor::div[@id='remediaLink'])]//text()").getall()
        text = " ".join(header + body).strip()
        slug = re.sub(r"[^a-z0-9]+", "-", response.meta["remedy"]).strip("-")
        (self.out_dir / f"{slug}.txt").write_text(text, encoding="utf-8")
        yield {"remedy": response.meta["remedy"], "chars": len(text)}
