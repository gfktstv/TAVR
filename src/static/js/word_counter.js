import { createToast } from "./toasts.js";
import { analyseEssay } from "./main.js";

function createToastAndTremble(buttonID) {
    createToast("notEnoughWords");
    const button = document.getElementById(buttonID);
    button.style.animation = "tremble 0.15s ease forwards";
    setTimeout(() => button.style.animation = "none", 150);
};

function countWords(text) {
    // Return 0 if the input is not a string or is empty after trimming.
    if (typeof text !== 'string' || text.trim() === '') {
      return 0;
    }
  
    /*
     * This regular expression is designed to find all "words" according to the specified rules.
     * It is composed of two main parts, separated by the OR operator (|).
     * The 'g' flag ensures all matches are found, and 'i' makes it case-insensitive (for AM/PM).
     *
     * Part 1: Matches standard words, including contractions and hyphenated words.
     * [a-zA-Z]+      - Matches a sequence of one or more letters.
     * (?:            - Starts a non-capturing group for hyphenated/contracted parts.
     *   [-’'][a-zA-Z]+ - Matches a hyphen, apostrophe, or curly apostrophe, followed by more letters.
     * )*             - The group can repeat zero or more times (e.g., for "up-to-date").
     *
     * Part 2: Matches numbers, dates, and times.
     * \d+            - Matches a sequence of one or more digits (e.g., "55", "30000").
     * (?:            - Starts a non-capturing group for separators like ',', '.', ':'.
     *   [.,:]\d+      - Matches a comma, dot, or colon followed by more digits.
     * )*             - The group can repeat (e.g., for "12.06.2016").
     * (?:st|nd|rd|th)? - Optionally matches ordinal suffixes (e.g., "1st", "2nd").
     * (?:[ap]m)?     - Optionally matches "am" or "pm" (case-insensitive due to 'i' flag).
     */
    const wordRegex = /[a-zA-Z]+(?:[-’'][a-zA-Z]+)*|\d+(?:[.,:]\d+)*(?:st|nd|rd|th)?(?:[ap]m)?/gi;
  
    const matches = text.match(wordRegex);
    return matches ? matches.length : 0;
};

function validateEssayLength() {
    let text = this.value.trim()
    let matches = countWords(text)

    let buttonAnalyse = document.getElementById("buttonAnalyse");
    let buttonWords = document.getElementById("buttonWords");

    buttonWords.textContent = matches + " Words";
    
    if (matches >= 250 && matches <= 350) {
        buttonAnalyse.style.backgroundColor = "#2670FF";
        buttonAnalyse.style.boxShadow = "0px 0px 10px #2670FF30";
        buttonAnalyse.onmouseover = function() {
            buttonAnalyse.style.backgroundColor = "#004EE5";
            buttonAnalyse.style.transition = "0.45s";
        };
        buttonAnalyse.onmouseleave = function() {
            buttonAnalyse.style.backgroundColor = "#2670FF";
            buttonAnalyse.style.transition = "0.45s";
        };

        buttonAnalyse.onclick = () => analyseEssay();
        buttonWords.onclick = null;

        buttonWords.disabled = true;
        buttonWords.style.cursor = "default";
    } else {
        buttonAnalyse.style.backgroundColor = "#BCBCBC";
        buttonAnalyse.style.boxShadow = "none";
        buttonAnalyse.onmouseover = function() {
            buttonAnalyse.style.backgroundColor = "#BCBCBC";
            buttonAnalyse.style.transition = "0.45s";
        };
        buttonAnalyse.onmouseleave = function() {
            buttonAnalyse.style.backgroundColor = "#BCBCBC";
            buttonAnalyse.style.transition = "0.45s";
        };

        buttonAnalyse.onclick = () => createToastAndTremble('buttonAnalyse');
        buttonWords.onclick = () => createToastAndTremble('buttonWords');

        buttonWords.disabled = false;
        buttonWords.style.cursor = "pointer";
    };
};

document.addEventListener("DOMContentLoaded", () => {
    let inputEssayArea = document.getElementById("inputEssayArea");
    validateEssayLength.call(inputEssayArea);
    inputEssayArea.addEventListener("input", validateEssayLength);
});