import argparse
import os
from urllib.request import urlretrieve
from sickle import Sickle

def download_image(image_url, output_dir):
      image_id = image_url.split("/")[-1]
      if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
      urlretrieve(image_url, f"{output_dir}/{image_id}.jpeg")

def main(args):
      sickle = Sickle('https://digital.ub.uni-duesseldorf.de/oai')
      ns = {'mets': 'http://www.loc.gov/METS/',
              'xlink': 'http://www.w3.org/1999/xlink'}

      record = sickle.GetRecord(identifier=f"oai:digital.ub.uni-duesseldorf.de:{args.vlid}", metadataPrefix='mets')

      for xml_element in record.xml.xpath('.//mets:fileGrp[@USE="MAX"]/mets:file/mets:FLocat/@xlink:href', namespaces=ns):
            download_image(xml_element, args.output)

if __name__ == "__main__":
      parser = argparse.ArgumentParser(description="Download images from Digitale Sammlungen ULB Düsseldorf using VL ID.")
      parser.add_argument("vlid", help="VL ID to download images for.")
      parser.add_argument("--output", default="output", help="Output directory for images.")
      args = parser.parse_args()
      main(args)