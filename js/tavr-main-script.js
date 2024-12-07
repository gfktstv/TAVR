// Adjust appropriate height of the page
document.body.clientHeight = document.body.clientHeight - 31 + 'px';
const mainContainer = document.getElementsByClassName('main-container')[0];
mainContainer.style.height = document.body.clientHeight - 31 + 'px';

// Navigation contacts button
function getContacts () {
    navigator.clipboard.writeText('g.feoktistoff@gmail.com').then(() => {
        console.log('Email copied to clipboard');
      },() => {
        console.error('Failed to copy email');
      });
    createToast('email');
}

// Main character counter
document.getElementById('inputEssayArea').addEventListener('input', function() {
    const textareaValue = this.value.trim();
    const buttonAnalyse = document.getElementById('buttonAnalyse');
    const buttonCharacters = document.getElementById('buttonCharacters');
    if (textareaValue.length >= 1500) {
        buttonAnalyse.style.backgroundColor = '#2670FF';
        buttonAnalyse.style.boxShadow = '0px 0px 10px #2670FF30';
        buttonAnalyse.onmouseover = function() {
            buttonAnalyse.style.backgroundColor = '#004EE5';
            buttonAnalyse.style.transition = '0.45s';
        };
        buttonAnalyse.onmouseleave = function() {
            buttonAnalyse.style.backgroundColor = '#2670FF';
            buttonAnalyse.style.transition = '0.45s';
        };
        buttonAnalyse.setAttribute('onclick', 'analyseEssay()');

        buttonCharacters.disabled = true;
        buttonCharacters.style.cursor = 'default';
    } else {
        buttonAnalyse.style.backgroundColor = '#BCBCBC';
        buttonAnalyse.style.boxShadow = 'None';
        buttonAnalyse.onmouseover = function() {
            buttonAnalyse.style.backgroundColor = '#BCBCBC';
            buttonAnalyse.style.transition = '0.45s';
        };
        buttonAnalyse.onmouseleave = function() {
            buttonAnalyse.style.backgroundColor = '#BCBCBC';
            buttonAnalyse.style.transition = '0.45s';
        };
        buttonAnalyse.setAttribute('onclick', 'createToastAndTremble(`buttonAnalyse`)');

        buttonCharacters.disabled = false;
        buttonCharacters.style.cursor = 'pointer';
    }
    buttonCharacters.textContent = textareaValue.length + ' Characters';
  });

// Not enough characters button click action
function createToastAndTremble(buttonID) {
    createToast("characters");
    const button = document.getElementById(buttonID);
    button.style.animation = 'tremble 0.15s ease forwards';
    setTimeout(() => button.style.animation = 'none', 150);
}

// Essay analysis
function analyseEssay() {
    // Button animation
    const button = document.getElementById(`buttonAnalyse`);
    button.style.animation = 'scale 0.15s ease forwards';
    setTimeout(() => button.style.animation = 'none', 150);
    // Add tables, chart and replaces textarea with marked up essay
    const inputEssayArea = document.getElementById(`inputEssayArea`);
    inputEssayArea.disabled = true;
    inputEssayArea.style.cursor = 'wait';
    getMeasurementsFromTAVR(inputEssayArea);
}

// Add tables, chart and replace textarea with marked up essay
async function getMeasurementsFromTAVR(inputEssayArea) {
    // Change cursor view to show loading
    document.body.style.cursor = 'wait';
    // Send a data to TAVR and get a response
    const essay = inputEssayArea.value;
    const url = 'http://localhost:5000/get_tables';
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ data: essay })
        });
        if (!response.ok) {
            throw new Error('Failed to send data to Flask server');
        }
        // The response is 4 tables and level in json format
        const measurements = await response.json();
        // Get tokens 
        const tokens_and_keys = await getTokens();
        // Add marked up essay and then add analytics using callback function
        addMarkedUpEssay(tokens_and_keys, function() {
          // Add charts and tables to quick analytics nav
          addMeasurementsToQckAnltcs(measurements['table_trigrams'], measurements['table_stats'], 
                                     measurements['table_recurring_lemmas'], measurements['level']);
          // Add charts and tables to analytics
          addMeasurementsToAnltcs(measurements['table_trigrams'], measurements['table_stats'], 
                                  measurements['table_academic_formulas'], measurements['table_academic_collocations'], 
                                  measurements['table_academic_words'], measurements['table_recurring_lemmas'], 
                                  measurements['level'], measurements['recurring_lemmas'], 
                                  measurements['len_academic_formulas'], measurements['len_academic_collocations'],
                                  measurements['len_academic_words']);
          // Animations of appearance
          // Hide horizontal scrollbar for a while
          document.getElementsByTagName(`body`)[0].style.overflowX = `hidden`;
          setTimeout(() => document.getElementsByTagName(`body`)[0].style.overflowX = ``, 1600);
          // Hide container with input essay and remove it
          const containerInputEssay = document.getElementById('containerInputEssay');
          containerInputEssay.style.animation = 'slideAway 0.55s ease-out forwards';
          setTimeout(() => containerInputEssay.remove(), 550);
          // Show marked up essay
          const containerMarkedUpEssay = document.getElementById('containerMarkedUpEssay');
          setTimeout(() => containerMarkedUpEssay.style.display = 'block', 450);
          setTimeout(() => containerMarkedUpEssay.style.animation = 'slideHere 0.55s ease-out forwards', 450);
          setTimeout(() => containerMarkedUpEssay.style.animation = 'none', 1000);
          // Quick analytics vocabulary chart appearance
          const vocabularyChartContainerQckAnltcs = document.getElementById('vocabularyChartContainerQckAnltcs');
          setTimeout(() => vocabularyChartContainerQckAnltcs.style.maxHeight = `300px`, 1150);
          // Quick analytics div and nav appears with slide1 animation
          const quickAnalytics = document.getElementById(`quickAnalytics`);
          setTimeout(() => quickAnalytics.style.display = `block`, 550);
          setTimeout(() => quickAnalytics.style.animation = `slide1 0.55s ease forwards`, 550);
          setTimeout(() => quickAnalytics.style.animation = `none`, 1150);
          // Quick analytics nav elements apear with slide2 animation
          const quickAnalyticsElements = quickAnalytics.getElementsByTagName(`li`);
          for (let i = 0; i < quickAnalyticsElements.length; i++ ) {
              quickAnalyticsElements[i].style.animation = `slide2 0.${5 + i}5s ease forwards`;
          }
          // Level and stats container appearance
          const levelAndStatsContainer = document.getElementById('levelAndStatsContainer');
          setTimeout(() => levelAndStatsContainer.style.maxHeight = `80px`, 1600);
          // Instruments nav appears with slide3 animation
          const instrumentsContainer = document.getElementById(`instrumentsContainer`);
          setTimeout(() => instrumentsContainer.style.display = `flex`, 550);
          setTimeout(() => instrumentsContainer.style.animation = `slide3 0.55s ease forwards`, 550);
          setTimeout(() => instrumentsContainer.style.animation = `none`, 1150);
          // Analytics appearance
          const containerAnalytics = document.getElementById('containerAnalytics');
          setTimeout(() => containerAnalytics.style.display = 'block', 1700);
          // Change cursor view to default
          document.body.style.cursor = 'default';
          inputEssayArea.style.cursor = 'default';
          // Change page height to hide blank places at the time of animations
          setTimeout(() => document.body.style.height = document.body.clientHeight + 500 + 'px', 350);
          setTimeout(() => instrumentsContainer.style.height = document.body.clientHeight + 500 + 'px', 350);
          setTimeout(() => quickAnalytics.style.height = document.body.clientHeight + 500 + 'px', 350);
          // Change page height to actual size
          setTimeout(() => document.body.style.height = containerMarkedUpEssay.clientHeight + containerAnalytics.clientHeight + 184 + 'px', 1701);
          setTimeout(() => instrumentsContainer.style.height = containerMarkedUpEssay.clientHeight + containerAnalytics.clientHeight + 184 + 'px', 1701);
          setTimeout(() => quickAnalytics.style.height = containerMarkedUpEssay.clientHeight + containerAnalytics.clientHeight + 184 + 'px', 1701);
        });
    } catch (error) {
        // Change cursor view to default
        document.body.style.cursor = 'default';
        inputEssayArea.style.cursor = 'default';
        // Change textarea disabled param
        const inputEssayArea = document.getElementById('inputEssayArea');
        inputEssayArea.disabled = false;
        // Show error message
        createToast(`connection`);
        console.log(error);
    }
  }

// Get marked up tokens from TAVR (returns both tokens in json format and array of tokens keys)
async function getTokens() {
    const url = 'http://localhost:5000/get_tokens';
    try {
        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
      });
      if (!response.ok) {
          throw new Error('Failed to send data to Flask server');
      }
      return await response.json();
    } catch (error) {
      console.log(error)
    }
  }

// Mark up input essay and add it to the web page
function addMarkedUpEssay(tokens_and_keys, _callback) {
    const markedTextArray = markUpText(tokens_and_keys);
    const inputMarkedEssay = document.getElementById('markedUpEssayArea');
    // Add marked up words
    for (let i = 0; i < markedTextArray.length; i++) {
        inputMarkedEssay.innerHTML += markedTextArray[i];
        inputMarkedEssay.innerHTML += ` `;
    }
    const mainContainer = document.getElementsByTagName('main')[0];
    mainContainer.style.paddingTop = '24px';
    _callback()
  }

// Create an array with span objects (marked up essay)
function markUpText(tokens_and_keys) {
    const tokens = tokens_and_keys['0'];
    const keys = tokens_and_keys['1'];
  
    // Create a new array to hold the marked up words
    const markedTextArray = [];
  
    markedTextArray.push(`<p>`)
    // Iterate through each word
    for (let i = 0; i < keys.length; i++) {
        // Create a span element for each word
        let span = document.createElement("span");
        span.textContent = keys[i];
        span.id = tokens[keys[i]]['id'];
        span.id = i;
        if (tokens[keys[i]]['functional_word'] || tokens[keys[i]]['punct']) {
            span.classList.add('no_data');
        } else {
            span.classList.add(tokenCLassFromLevel(tokens[keys[i]]['level']));
            span.setAttribute('onclick', 'getReplacements("' + span.id + '")');
        }
        if (keys[i] === '\n\n') {
            // Add the end of the paragraph and a start of a new paragraph to the markedWords array
            markedTextArray.push(`</p><p>`);
        } else {
            // Add the span element to the markedWords array
            markedTextArray.push(span.outerHTML);
        }
    }
    markedTextArray.push(`</p>`);
  
    return markedTextArray
}
  
// Get ID for span object from level of vocabulary
function tokenCLassFromLevel(level) {
    let result;
    if (level === 'A1') {
        result = 'level0';
    } else if (level === 'A2') {
        result = 'level1';
    } else if (level === 'B1') {
        result = 'level2';
    } else if (level === 'B2') {
        result = 'level3';
    } else if (level === 'C1') {
        result = 'level4';
    } else if (level === 'C2') {
        result = 'level5';
    } else {
        result = 'no_data';
    }
    return result
}

// Add measurments to quick analytics
function addMeasurementsToQckAnltcs(tableTrigrams, tableStats, tableRecurringLemmas, level) {
    const vocabularyChartContainerQckAnltcs = document.getElementById('vocabularyChartContainerQckAnltcs');
    vocabularyChartContainerQckAnltcs.innerHTML = `<img class='vocabulary-chart-quick-analytics' src='temporary_files/vocabulary_chart_qck_anltcs.png'>`;
    console.log(vocabularyChartContainerQckAnltcs.innerHTML);

    const recurringWordsContainerQckAnltcs = document.getElementById('recurringWordsContainerQckAnltcs');
    recurringWordsContainerQckAnltcs.innerHTML = tableRecurringLemmas;

    const frequentPhrasesContainerQckAnltcs = document.getElementById('frequentPhrasesContainerQckAnltcs');
    frequentPhrasesContainerQckAnltcs.innerHTML = tableTrigrams;

    const levelSpan = document.getElementById('levelSpan');
    levelSpan.innerHTML = level;
    if (level==='A1' || level==='A2') {
        levelSpan.style.backgroundImage = 'linear-gradient(to right top, #FFCF32, #FFE89C)'
    } else if (level==='B1' || level==='B2') {
        levelSpan.style.backgroundImage = 'linear-gradient(to right top, #0f60fd, #8fb9fd)'
    } else if (level==='C1' || level==='C2') {
        levelSpan.style.backgroundImage = 'linear-gradient(to right top, #6B66FF, #9C99FF)'
    }
    const statsInputContainer = document.getElementById('statsInputContainer');
    statsInputContainer.innerHTML = tableStats;
}

// Add measurements to analytics
function addMeasurementsToAnltcs(tableTrigrams, tableStats, tableAcademicFormulas, tableAcademicCollocations, 
                                 tableAcademicWords, tableRecurringLemmas, level, recurring_lemmas, 
                                 len_academic_formulas, len_academic_collocations, len_academic_words) {
    const vocabularyChartAnalytics = document.getElementById('vocabularyChartAnalytics');
    vocabularyChartAnalytics.src = 'temporary_files/vocabulary_chart_anltcs.png';

    const recurringLemmasTableContainerAnltcs = document.getElementById('recurringLemmasTableContainerAnltcs');
    recurringLemmasTableContainerAnltcs.innerHTML = tableRecurringLemmas;

    const recurringLemma1 = document.getElementById('recurringLemma1');
    recurringLemma1.innerHTML = recurring_lemmas[0];
    const recurringLemma2 = document.getElementById('recurringLemma2');
    recurringLemma2.innerHTML = recurring_lemmas[1];
    const recurringLemma3 = document.getElementById('recurringLemma3');
    recurringLemma3.innerHTML = recurring_lemmas[2];

    if (len_academic_formulas > 1 || len_academic_words > 1 || len_academic_collocations > 1) {
        const academicFormulasTablesContainerAnltcs = document.getElementById('academicFormulasTablesContainerAnltcs');
        academicFormulasTablesContainerAnltcs.innerHTML += tableAcademicFormulas;
        academicFormulasTablesContainerAnltcs.innerHTML += tableAcademicCollocations;
        academicFormulasTablesContainerAnltcs.innerHTML += tableAcademicWords;
    } else {
        // Delete analytics sub-container with academic formulas
        const academicFormulasContainerAnltcs = document.getElementById('academicFormulasContainerAnltcs');
        academicFormulasContainerAnltcs.remove()
        // Change row direction for next analytics sub-containers
        const trigramsContainerAnltcs = document.getElementById('trigramsContainerAnltcs');
        trigramsContainerAnltcs.style.flexDirection = 'row-reverse';
    }

    const trigramsTableContainerAnltcs = document.getElementById('trigramsTableContainerAnltcs');
    trigramsTableContainerAnltcs.innerHTML = tableTrigrams;
}

// Define instruments checkboxes action
function changeLevelHighlighting(buttonID, level, borderColor, backColor) {
    const levelButton = document.getElementById(buttonID);
    const levelSpans = document.getElementsByClassName(level);
    if (levelButton.checked === true) {
        for (let i = 0; i < levelSpans.length; i++) {
            levelSpans[i].style.border = '1px solid ' + borderColor;
            levelSpans[i].style.backgroundColor = backColor;
        }
    } else {
        for (let i = 0; i < levelSpans.length; i++) {
            levelSpans[i].style.border = '1px solid #DADADA';
            levelSpans[i].style.backgroundColor = '#f9f9f982';
        }
    }
}
function checkUncheck() {
    let checked_value;
    const button = document.getElementById('checkUncheckButtonInstruments');
    const instrumentsContainer = document.getElementById(`instrumentsContainer`);
    const instrumentsCheckboxes = instrumentsContainer.getElementsByTagName('input');
    if (button.innerText === 'Uncheck all') {
        checked_value = false;
        button.innerText = 'Check all';
    } else {
        checked_value = true;
        button.innerText = 'Uncheck all';
    }
    for (let i = 0; i < instrumentsCheckboxes.length; i++) {
        console.log(instrumentsCheckboxes[i].checked);
        console.log(checked_value);
        if (checked_value !== instrumentsCheckboxes[i].checked) {
            instrumentsCheckboxes[i].click();
        }
    }
}

// Copy button action
function copyEssay() {
  const innerText = document.getElementById('markedUpEssayArea').innerText;
  const regexPunct = /(\s)(?<punct>[^-\w\s]+)/gi;
  let text = innerText.replace(regexPunct, '$<punct>');
  const regexDash = /(\s)([-]+)(\s)/gi;
  text = text.replace(regexDash, '-');
  navigator.clipboard.writeText(text).then(() => {
    createToast(`essayCopy`);
    console.log('The essay with the changes made copied');
  },() => {
    createToast(`essayCopyError`);
    console.error('Failed to copy');
  });
}

// Start over button action
function startOver() {
  if (confirm("Do you want to clear the essay and the stats and start over?")) {
    location.reload();
  }
}

// Replacements
// Get replacement options by id of a token from TAVR
async function getReplacements(tokenID) {
    // Change cursor view to show loading
    document.body.style.cursor = 'progress';
    // Get response from TAVR
    const url = 'http://localhost:5000/get_replacements';
    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }, 
            body: JSON.stringify({ data: tokenID })
        });
        if (!response.ok) {
            throw new Error('Failed to send data to Flask server');
        }
        const replacementsAndLevels = await response.json();
        // Add pop up window
        addReplacementsPopUpWindow(tokenID, replacementsAndLevels['lemmas'], replacementsAndLevels['levels'])
        // Change cursor view to default
        document.body.style.cursor = 'default';
    } catch (error) {
        // Change cursor view to default
        document.body.style.cursor = 'default';
        console.log(error);
        
    }
}

function addReplacementsPopUpWindow(tokenID, lemmas, levels) {
    // Delete existing replacement window
    try {
        const replacementWindow = document.getElementById('replacementsPopUpWindow');
        replacementWindow.remove();
    } catch(error) {
        console.log('Failed to remove replacement window');
    }
    // Add new one
    const markedUpEssayArea = document.getElementById('markedUpEssayArea');
    const replacementsPopUpWindow = document.createElement('div');
    replacementsPopUpWindow.id = 'replacementsPopUpWindow';
    for (let i = 0; i < lemmas.length; i++) {
        const buttonReplacement = document.createElement('button');
        buttonReplacement.id = 'buttonReplacement';
        buttonReplacement.classList.add(`button-${tokenCLassFromLevel(levels[i])}`);
        buttonReplacement.textContent = lemmas[i];
        buttonReplacement.setAttribute(
            'onclick',
            'replace(' + '`' + tokenID + '`' + ', ' + '`' + lemmas[i] + '`' + ', ' + '`'+ levels[i] + '`' + ')'
        );

        replacementsPopUpWindow.innerHTML += buttonReplacement.outerHTML;
    }
    if (lemmas.length === 0) {
        const buttonReplacement = document.createElement('button');
        buttonReplacement.id = 'noReplacements';
        buttonReplacement.textContent = 'no replacements';
        
        replacementsPopUpWindow.innerHTML += buttonReplacement.outerHTML;
    }
    const position = document.getElementById(String(tokenID)).getBoundingClientRect();
    const x_pos = position.left;
    const y_pos = position.top + 30 + window.scrollY;
    replacementsPopUpWindow.style.position = 'absolute';
    replacementsPopUpWindow.style.left = x_pos + 'px';
    replacementsPopUpWindow.style.top = y_pos + 'px';

    replacementsPopUpWindow.setAttribute('onmouseleave', 'removeReplacementWindow(3000)');

    markedUpEssayArea.innerHTML += replacementsPopUpWindow.outerHTML;
}

function replace(tokenID, lemma, level) {
    const spanObject = document.getElementById(String(tokenID));
    spanObject.textContent = lemma;

    spanObject.classList.remove(`${spanObject.className}`);
    spanObject.classList.add(tokenCLassFromLevel(level));

    const replacementsPopUpWindow = document.getElementById('replacementsPopUpWindow');
    replacementsPopUpWindow.setAttribute('onmouseleave', 'removeReplacementWindow(1550)');
}

function removeReplacementWindow(timeout) {
    const replacementWindow = document.getElementById('replacementsPopUpWindow');
    setTimeout(() => replacementWindow.style.opacity = '0', timeout - 550);
    setTimeout(() => replacementWindow.remove(), timeout);
}

// Notifications
const toastDetails = {
    characters: {
        text: 'Please, enter more characters (at least 1500)'
    },
    email: {
        text: 'Email has been copied'
    },
    tokens: {
        text: 'TAVR has not recognized enough English words for analysis'
    },
    connection: {
        text: 'No connection with the TAVR server'
    },
    essayCopy: {
      text: 'The essay with the changes made has been copied'
    },
    essayCopyError: {
      text: 'Failed to copy :('
    }
}

const list = document.getElementById('notificationsList');
function createToast (className) {
    const toast = document.createElement('li');
    toast.className = `toast-${className}`;
    const text = toastDetails[className].text;
    toast.innerHTML = `<span class="${className}">${text}</span>`;
    list.appendChild(toast);
    removeToast(toast);
}

function removeToast(toast) {
    setTimeout(() => toast.style.animation = 'fadeOut 1.5s ease-out', 3500)
    setTimeout(() => toast.remove(), 5000);
}