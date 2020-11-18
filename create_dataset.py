from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementClickInterceptedException
import urllib.request
import os
import errno


characters = [
                'ryu', 'oro', 'dudley', 'elena', 'hugo', 'ken',
                'yun', 'remy', 'q', 'chun-li', 'makoto', 'twelve', 'yang', 'gill',
                'akuma', 'urien', 'necro', 'ibuki', 'sean', 'alex'
             ]
driver = webdriver.Chrome()


# Set up the webpage and elements given a character to download
def setup_driver(character):
    url = "https://www.justnopoint.com/zweifuss/{}/{}.htm".format(character, character)
    driver.get(url)

    # Get image
    img = driver.find_element_by_name('box') # image element to get gif from

    # Get character color palette buttons
    colors = driver.find_element_by_id('colors')
    colors = colors.find_elements(By.TAG_NAME, 'a')

    # Get categories of attacks
    attacks_table = driver.find_element_by_xpath('//*[@id="attacks"]/table/tbody')
    attacks_box = driver.find_element_by_xpath('//*[@id="attacks-box"]')
    attacks = attacks_box.text.replace(attacks_table.text, '')
    attacks = attacks.split('\n')
    attacks = attacks[1:] # remove ATTACKS header
    i = 0
    while i < len(attacks): # remove random newlines added by the site
        if not attacks[i]:
            attacks.pop(i)
        i += 1

    # Get categories of non-attacks
    non_attacks_table = driver.find_element_by_xpath('//*[@id="non-attacks"]/table/tbody')
    non_attacks_box = driver.find_element_by_xpath('//*[@id="non-attacks-box"]')
    non_attacks = non_attacks_box.text.replace(non_attacks_table.text, '')
    non_attacks = non_attacks.split('\n')
    non_attacks = non_attacks[1:] # remove NON-ATTACKS header
    i = 0
    while i < len(non_attacks): # remove random newlines added by the site
        if not non_attacks[i]:
            non_attacks.pop(i)
        i += 1

    return attacks, attacks_table, non_attacks, non_attacks_table, colors, img


# Cycle through attacks and download them
def download_moves(character, moves, table, color_num, img):
    rows = table.find_elements(By.TAG_NAME, 'tr')
    i = 0 # row counter
    for i in range(len(moves)):
        cols = rows[i].find_elements(By.TAG_NAME, 'td')
        # Some categories contain '/', which creates another directory by accident
        moves[i] = moves[i].replace('/', '+')
        folder = "{}/{}".format(character, moves[i])
        # Create folder if it doesn't already exist
        try:
            os.makedirs(folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        for col in cols:
            try:
                # Set width to 0 to stop it from covering other buttons
                driver.execute_script("document.getElementsByName('box')[0].style.width = 0;")
                move = col.find_element(By.TAG_NAME, 'a')
                move.click()
                img_src = img.get_attribute("src")
                try:
                    urllib.request.urlretrieve(img_src, "{}/color{}-{}.gif".format(folder, color_num, move.text))
                except:
                    print("Failed to download {}, color {}, {}".format(character, color_num, move, move.text))
            except NoSuchElementException:
                # Some elements aren't clickable, just pass
                pass


if __name__ == '__main__':
    for character in characters:
        # Don't download characters that already have folders
        if not os.path.isdir(character):
            attacks, attacks_table, non_attacks, non_attacks_table, colors, img = setup_driver(character)
        
            for i in range(len(colors)):
                colors[i].click()

                # Download attacks
                download_moves(character, attacks, attacks_table, i, img)

                # Download non-attacks
                rows = non_attacks_table.find_elements(By.TAG_NAME, 'tr')
                download_moves(character, non_attacks, non_attacks_table, i, img)