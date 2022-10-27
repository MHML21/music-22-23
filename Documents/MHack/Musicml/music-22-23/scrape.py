import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC




def scrape_chords(i, n, filename):
    """
    Scrapes the chords off n songs on ufret, starting at index i
    """
    with open(f'{filename}.csv', 'a', encoding='UTF8', newline='') as f:
        # set up selenium driver & csv writer
        writer = csv.writer(f)
        print("Loading Webdriver")
        driver = webdriver.Firefox()
        
        # scrape chords
        print("Starting Scrape")
        for i in range(i, i+n):
            driver.get(f"https://www.ufret.jp/song.php?data={i}")
            try:
                WebDriverWait(driver, 1).until(EC.presence_of_element_located((By.TAG_NAME, "rt"))) # wait until page loads
                writer.writerow([res.text for res in driver.find_elements(By.TAG_NAME, "rt")])      # save all chords as a single list
                print(i)
            except Exception:
                print("Timeout")
                continue
        driver.close()



if __name__ == '__main__':
    # scrapes chords off of ufret
    scrape_chords(6800, 30000, "data/rawdata")
    #https://medium.com/@huanlui/chordsuggester-i-3a1261d4ea9e
    #https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5