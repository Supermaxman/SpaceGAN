import scrapy
import os
import shutil

archive_url = 'https://www.spacetelescope.org/static/archives/images/original/'
image_folder = 'D:/Data/nasa/images'
output_folder = 'D:/Data/nasa/observations'


# TODO change timeout to longer than 180 seconds, change download warn size and max size to be ~2 gb
class NasaImageSpider(scrapy.Spider):
    name = 'nasaspider'
    start_urls = [
        f'https://www.spacetelescope.org/images/archive/search/page/{i}/' 
        # filtered by "observations" and by "fine" quality & minimum resolution "400x300"
        '?ranking=20&fov=0&release_id=&minimum_size=1&description=&published_until_year=0' 
        '&published_until_month=0&title=&subject_name=&credit=&published_until_day=0' 
        '&published_since_day=0&published_since_month=0&type=Observation&id=&published_since_year=0'
        for i in range(1, 44)
    ]
    custom_settings = {
        'DOWNLOAD_TIMEOUT': 1000,
        'DOWNLOAD_MAXSIZE': 0,
        'DOWNLOAD_WARNSIZE': 0
    }

    def parse(self, response):
        for script in response.css('div[class=col-md-12] script').extract():
            for line in script.split('\n'):
                line = line.strip()
                line = line.split(':')
                if len(line) > 1 and line[0] == 'url':
                    image_name = line[1].strip()
                    image_name = image_name.replace('\'', '').replace(',', '').replace('/images/', '').replace('/', '') + '.tif'
                    image_url = f'{archive_url}{image_name}'
                    image_path = os.path.join(image_folder, image_name)
                    if not os.path.exists(image_path):
                        yield response.follow(image_url, self.parse_image)
                    else:
                        output_path = os.path.join(output_folder, image_name)
                        if not os.path.exists(output_path):
                            shutil.copyfile(image_path, output_path)

    def parse_image(self, response):
        image_name = response.url.split('/')[-1]
        image_path = os.path.join(image_folder, image_name)
        if response.status == 200:
            if not os.path.exists(image_path):
                with open(image_path, 'wb') as f:
                    f.write(response.body)
                output_path = os.path.join(output_folder, image_name)
                if not os.path.exists(output_path):
                    shutil.copyfile(image_path, output_path)
        else:
            print(f'Skipping {image_name} (status={response.status})')
