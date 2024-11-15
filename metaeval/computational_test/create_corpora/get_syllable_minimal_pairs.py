"""
    @author María Andrea Cruz Blandón
    @date 01.11.2023

    This script do web scraping of the lexical resource of the Hong Kong University to extract the syllables that are
    minimal pairs of tones 2 (tone 25) and 3 (tone 33) in Cantonese.
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from typing import List, Tuple


def _get_syllable_minimal_pairs() -> List[Tuple[str, str]]:
    """
        It gets the minimal pairs for tone 25 and tone 33 from the available syllables in the Cantonese Language
    """
    url = "https://humanum.arts.cuhk.edu.hk/Lexis/lexi-mf/syllables.php"
    op = webdriver.ChromeOptions()
    op.add_argument('headless')
    driver = webdriver.Chrome(options=op)
    driver.get(url)

    syllables_table = driver.find_element(By.ID, "syllables_table")
    total_rows = len(syllables_table.find_elements(By.TAG_NAME, "tr"))
    total_cells = len(syllables_table.find_elements(By.TAG_NAME, "tr")[0].find_elements(By.TAG_NAME, "td"))

    syllables = []

    for i in range(2, total_rows):
        for j in range(2, total_cells + 2):
            cell = driver.find_elements(By.XPATH, f"//table[@id='syllables_table']/tbody/tr[{i}]/td[{j}]")[0]

            if cell.text == "":
                continue

            cell.find_element(By.TAG_NAME, "a").click()
            local_syllables_table = driver.find_element(By.CLASS_NAME, "pho-rel")
            local_rows = local_syllables_table.find_elements(By.TAG_NAME, "tr")
            local_syllables = []
            for local_row in local_rows[1:]:
                syllable = local_row.find_elements(By.TAG_NAME, "td")[0].text
                local_syllables.append(syllable)
            tones = [syllable[-1] for syllable in local_syllables]
            if '2' in tones and '3' in tones:
                syllables.append((f'{local_syllables[0][:-1]}2', f'{local_syllables[0][:-1]}3'))
            driver.back()
    driver.close()

    return syllables


with open('./cantonese_minimal_pairs.txt', 'a') as f:
    for pair in _get_syllable_minimal_pairs():
        f.write(f'{pair[0]}\t{pair[1]}\n')
