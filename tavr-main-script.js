// Character counter
document.getElementById('inputEssay').addEventListener('input', function() {
  var textareaValue = this.value.trim();
  var buttonAssess = document.getElementById('buttonAssess');
  var characterCounter = document.getElementById('characterCount')
  if (textareaValue.length >= 1200) {
      buttonAssess.disabled = false;
      characterCounter.style.color='#1400FF'
  } else {
      buttonAssess.disabled = true;
      characterCounter.style.color='#ff0000'
  }

  var characterCount = document.getElementById('characterCount');
  characterCount.textContent = textareaValue.length;
});

// Change the view of the web page after click on "Analyze"
function showAssessment() {
  // Disable textarea
  const textArea = document.getElementById('inputEssay');
  textArea.disabled = true;
  // Disable button 
  const buttonAssess = document.getElementById('buttonAssess');
  buttonAssess.disabled = true;
  // Add measurements from TAVR
  getMeasurementsFromTAVR(inputEssay);
};

// Add tables, chart and replace textarea with marked up essay
async function getMeasurementsFromTAVR(inputEssay) {
  // Change cursor view to show loading
  document.body.style.cursor = 'wait'
  // Send a data to TAVR and get a response
  const essay = inputEssay.value
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
      // The response is 3 tables in json format
      const tables = await response.json();
      // Add chart and tables to the webpage
      addMeasurements(tables['0'], tables['1'], tables['2']);
      // The response is tokens in json format and tokens keys in array format (because in json tokens are sorted and idk why)
      var tokens_and_keys = await getTokens();
      // Add marked up essay and remove textarea
      addMarkedUpEssay(tokens_and_keys)
      inputEssay.remove()
      // Change title
      document.getElementById('title-above-essay').innerHTML = 'the Tool for Analysis of Vocabulary Richness';
      // Remove the buttonAssess and add a buttonClear and a buttonCopy
      const buttonAssess = document.getElementById('buttonAssess');
      buttonAssess.remove();
      const characterCount = document.getElementById('characterCount');
      characterCount.remove();
      const inputButton = document.getElementById('inputButton');
      inputButton.innerHTML += `
        <button id="buttonCopy" onclick="copyText()"><strong>Copy</string></button>
        <button id="buttonClear" onclick="clearText()"><strong>Clear</strong></button>
      `;
      // Make a smooth scroll a little bit lower
      scrollTo({
        top: 100,
        behavior: "smooth"
      })
      // Change cursor view to default
      document.body.style.cursor = 'default'
  } catch (error) {
      // Change cursor view to default
      document.body.style.cursor = 'default'
      // Change textarea disabled param
      const textArea = document.getElementById('inputEssay');
      textArea.disabled = false;
      alert('An error occurred, please, try again')
      console.log(error)
  }
}

// Mark up input essay and add it to the web page
function addMarkedUpEssay(tokens_and_keys) {
  var markedTextArray = markUpText(tokens_and_keys);
  const inputMarkedEssay = document.getElementById('inputMarkedEssay');
  inputMarkedEssay.id = 'markedEssay';
  // Add marked up words
  for (var i = 0; i < markedTextArray.length; i++) {
    inputMarkedEssay.innerHTML += markedTextArray[i]
    inputMarkedEssay.innerHTML += ` `
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
    const tokens = await response.json();
    return tokens;
  } catch (error) {
    console.log(error)
  }
}

// Add tables, chart and level to the web page
function addMeasurements(tableTrigrams, tableStats, tableAcademicFormulas) {
  const inputContainerAssessment = document.getElementById('inputContainerAssessment');
  inputContainerAssessment.innerHTML += `
    <div class='containerAssessment'>
      <table class='tableAssessment'>
        <tr class='rowAssessment'>
          <td id='cellChartAndStats' class='cellAssessment'></td>
          <td id='cellTrigrams' class='cellAssessment'></td>
          <td id='cellLevelAndAcademicFormulas' class='cellAssessment'></td>
        </tr>
      </table>
    </div>
  `;
  const cellChartAndStats = document.getElementById('cellChartAndStats');
  cellChartAndStats.innerHTML += `<img class='vocabularyChart', src='temporary_files/vocabulary_chart.png'>`;
  cellChartAndStats.innerHTML += tableStats;

  const cellTrigrams = document.getElementById('cellTrigrams');
  cellTrigrams.innerHTML = tableTrigrams;

  const cellLevelAndAcademicFormulas = document.getElementById('cellLevelAndAcademicFormulas');
  cellLevelAndAcademicFormulas.innerHTML = tableAcademicFormulas;
}

// Create an array with span objects (marked up essay)
function markUpText(tokens_and_keys) {
  var tokens = tokens_and_keys['0'];
  var keys = tokens_and_keys['1'];

  // Create a new array to hold the marked up words
  var markedTextArray = [];

  // Iterate through each word
  for (var i = 0; i < keys.length; i++) {
      // Create a span element for each word
      var span = document.createElement("span");
      span.textContent = keys[i];
      span.id = tokens[keys[i]]['id'];
      span.id = i;
      if (tokens[keys[i]]['stopword']) {
        span.classList.add('no_data');
      } else {
        span.classList.add(tokenCLassFromLevel(tokens[keys[i]]['level']));
        span.setAttribute('onclick', 'getReplacements("' + span.id + '")');
      }
      // Add the span element to the markedWords array
      markedTextArray.push(span.outerHTML);
  }

  return markedTextArray
}

// Get ID for span object from level of vocabulary
function tokenCLassFromLevel(level) {
  if (level === 'A1') {
    result = 'level0'
  } else if (level === 'A2') {
    result = 'level1'
  } else if (level === 'B1') {
    result = 'level2'
  } else if (level === 'B2') {
    result = 'level3'
  } else if (level === 'C1') {
    reuslt = 'level4'
  } else if (level === 'C2') {
    result = 'level5'
  } else {
    result = 'no_data'
  }
  return result
}

// Get replacement options by id of a token from TAVR
async function getReplacements(tokenID) {
  // Change cursor view to show loading
  document.body.style.cursor = 'progress'
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
      const replacements = await response.json();
      console.log(replacements)
      // Add pop up window
      addReplacementsPopUpWindow(tokenID, replacements)
      // Change cursor view to default
      document.body.style.cursor = 'default'
  } catch (error) {
      // Change cursor view to default
      document.body.style.cursor = 'default'
      console.log(error)
      
  }
}

function addReplacementsPopUpWindow(tokenID, replacements) {
  // Delete existing replacement window
  try {
    const replacementWindow = document.getElementById('replacementsPopUpWindow');
    replacementWindow.remove()
  } catch(error) {
    
  }
  const markedEssay = document.getElementById('markedEssay');
  const replacementsPopUpWindow = document.createElement('div');
  replacementsPopUpWindow.id = 'replacementsPopUpWindow';
  for (var i = 0; i < replacements.length; i++) {
    const buttonReplacement = document.createElement('button');
    buttonReplacement.id = 'buttonReplacement';
    buttonReplacement.textContent = replacements[i];
    buttonReplacement.setAttribute('onclick', 'replace(' + '`' + tokenID + '`' + ', ' + '`' + replacements[i] + '`' + ')');

    replacementsPopUpWindow.innerHTML += buttonReplacement.outerHTML;
  };
  if (replacements.length === 0) {
    const buttonReplacement = document.createElement('button');
    buttonReplacement.id = 'noReplacements';
    buttonReplacement.textContent = 'no replacements';

    replacementsPopUpWindow.innerHTML += buttonReplacement.outerHTML;
  }
  const position = document.getElementById(String(tokenID)).getBoundingClientRect();
  console.log(position)
  const x_pos = position.left;
  const y_pos = position.top + 35 + window.scrollY;
  replacementsPopUpWindow.style.position = 'absolute';
  replacementsPopUpWindow.style.left = x_pos + 'px';
  replacementsPopUpWindow.style.top = y_pos + 'px';

  replacementsPopUpWindow.setAttribute('onmouseleave', 'removeReplacementWindow()')

  markedEssay.innerHTML += replacementsPopUpWindow.outerHTML;
}

function replace(tokenID, replacement) {
  const spanObject = document.getElementById(String(tokenID));
  spanObject.textContent = replacement;

  spanObject.classList.remove(spanObject.class)
  spanObject.classList.add('replaced');
}

function removeReplacementWindow() {
  const replacementWindow = document.getElementById('replacementsPopUpWindow')
  replacementWindow.remove()
}

// Buttons "clear" and "copy"

function clearText() {
  if (confirm("Do you want to clear the essay and the stats?")) {
    location.reload()
  }
}

function copyText() {
  const inputEssay = document.getElementById('inputEssay').value
  navigator.clipboard.writeText(inputEssay).then(() => {
    console.log('Content copied to clipboard');
    /* Resolved - text copied to clipboard successfully */
  },() => {
    console.error('Failed to copy');
    /* Rejected - text failed to copy to the clipboard */
  });
}