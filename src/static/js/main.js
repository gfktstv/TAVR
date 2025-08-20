import { ANALYZE_ESSAY_ENDPOINT, GET_REPLACEMENTS_ENDPOINT } from "./constants.js";
import { createToast } from "./toasts.js";

export async function analyseEssay() {
    // Button animation
    const button = document.getElementById(`buttonAnalyse`);
    button.style.animation = "scale 0.15s ease forwards";
    setTimeout(() => button.style.animation = "none", 150);
    // Add tables, chart and replace textarea with marked up essay
    const inputEssayArea = document.getElementById(`inputEssayArea`);

    const url = `${ANALYZE_ESSAY_ENDPOINT}`;
    const essayText = inputEssayArea.value;

    let analytics;
    try {
        const analyticsResponse = await fetch(url, 
            {
                method: "POST", 
                headers: {"Content-Type": "application/json"}, 
                body: JSON.stringify({ text: essayText})
            });
        analytics = await analyticsResponse.json();
    } catch(error) {
        createToast(`noConnectionWithServer`);
        console.log(error);
        return;
    };

    await addMarkedUpEssay(analytics["marked_up_tokens"]);
    addMeasurementsToQckAnltcs({
        trigramsTable: analytics.trigrams_table,
        statsTable: analytics.stats_table,
        recurringLemmasTable: analytics.recurring_lemmas_table,
        level: analytics.level
    });
    addMeasurementsToAnltcs({
        trigramsTable: analytics.trigrams_table,
        statsTable: analytics.stats_table,
        academicFormulasTable: analytics.academic_formulas_table,
        academicCollocationsTable: analytics.academic_collocations_table,
        academicWordsTable: analytics.academic_words_table,
        recurringLemmasTable: analytics.recurring_lemmas_table,
        level: analytics.level,
        recurringLemmas: analytics.recurring_lemmas,
        lenAcademicFormulas: analytics.len_academic_formulas_table,
        lenAcademicCollocations: analytics.len_academic_collocations_table,
        lenAcademicWords: analytics.len_academic_words_table
    });
    runAnimations();
}

function runAnimations() {
    const body = document.body;
    const containerInputEssay = document.getElementById("containerInputEssay");
    const containerMarkedUpEssay = document.getElementById("containerMarkedUpEssay");
    const vocabularyChartContainerQckAnltcs = document.getElementById("vocabularyChartContainerQckAnltcs");
    const quickAnalytics = document.getElementById("quickAnalytics");
    const levelAndStatsContainer = document.getElementById("levelAndStatsContainer");
    const instrumentsContainer = document.getElementById("instrumentsContainer");
    const containerAnalytics = document.getElementById("containerAnalytics");

    const TIMINGS = {
        slideDuration: 550,

        showEssay: 450,

        quickAnalyticsStart: 550,
        quickAnalyticsEnd: 1150,

        levelStats: 1600,

        instrumentsStart: 550,
        instrumentsEnd: 1150,

        analytics: 1700
    };    

    // 1. Hide horizontal scrollbar temporarily
    applyStyleLater(body, "overflowX", "hidden", 0);
    applyStyleLater(body, "overflowX", "", 1600);

    // 2. Transition essay containers
    containerInputEssay.style.animation = "slideAway 0.55s ease-out forwards";
    setTimeout(() => containerInputEssay.remove(), TIMINGS.slideDuration);

    applyStyleLater(containerMarkedUpEssay, "display", "block", TIMINGS.showEssay);
    applyStyleLater(containerMarkedUpEssay, "animation", "slideHere 0.55s ease-out forwards", TIMINGS.showEssay);
    applyStyleLater(containerMarkedUpEssay, "animation", "none", TIMINGS.showEssay + TIMINGS.slideDuration);

    // 3. Quick analytics
    applyStyleLater(quickAnalytics, "display", "block", TIMINGS.quickAnalyticsStart);
    applyStyleLater(quickAnalytics, "animation", "slide1 0.55s ease forwards", TIMINGS.quickAnalyticsStart);
    applyStyleLater(quickAnalytics, "animation", "none", TIMINGS.quickAnalyticsEnd);

    applyStyleLater(vocabularyChartContainerQckAnltcs, "maxHeight", "300px", TIMINGS.quickAnalyticsEnd);

    const quickAnalyticsElements = quickAnalytics.getElementsByTagName("li");
    for (let i = 0; i < quickAnalyticsElements.length; i++) {
        quickAnalyticsElements[i].style.animation = `slide2 0.${5 + i}5s ease forwards`;
    }

    // 4. Level & stats
    applyStyleLater(levelAndStatsContainer, "maxHeight", "80px", TIMINGS.levelStats);

    // 5. Instruments nav
    applyStyleLater(instrumentsContainer, "display", "flex", TIMINGS.instrumentsStart);
    applyStyleLater(instrumentsContainer, "animation", "slide3 0.55s ease forwards", TIMINGS.instrumentsStart);
    applyStyleLater(instrumentsContainer, "animation", "none", TIMINGS.instrumentsEnd);

    // 6. Analytics section
    applyStyleLater(containerAnalytics, "display", "block", TIMINGS.analytics);

    // 7. Adjust page height dynamically
    setTimeout(() => {
        const newHeight = body.clientHeight + 300 + "vh";
        body.style.height = newHeight;
        instrumentsContainer.style.height = newHeight;
        quickAnalytics.style.height = newHeight;
    }, 0);

    setTimeout(() => {
        const actualHeight = containerMarkedUpEssay.clientHeight + containerAnalytics.clientHeight + 280 + "px";
        body.style.height = actualHeight;
        instrumentsContainer.style.height = actualHeight;
        quickAnalytics.style.height = actualHeight;
    }, TIMINGS.analytics + 1);
}


function applyStyleLater(element, property, value, delay) {
    setTimeout(() => {
        element.style[property] = value;
    }, delay);
};

function addMarkedUpEssay(marked_up_tokens) {
    return new Promise((resolve) => {
        const markedTextArray = markUpText(marked_up_tokens);
        const inputMarkedEssay = document.getElementById("markedUpEssayArea");

        for (let i = 0; i < markedTextArray.length; i++) {
            inputMarkedEssay.innerHTML += markedTextArray[i] + " ";
        };

        resolve();
    });
};

function markUpText(marked_up_tokens) {
    const markedTextArray = [];
    markedTextArray.push("<p>");

    for (const [tokenID, token] of Object.entries(marked_up_tokens)) {
        if (token.text === "\n\n") {
            // Close + open paragraph on double line break
            markedTextArray.push("</p><p>");
            continue;
        }

        markedTextArray.push(
            `<span id="${tokenID}" class="${tokenClassFromLevel(token.level)}" data-id="${token.id}" data-pos="${token.pos}">${token.text}</span>`
          );
    }

    markedTextArray.push("</p>");      
    return markedTextArray;
};

function addMeasurementsToQckAnltcs({
    trigramsTable,
    statsTable,
    recurringLemmasTable,
    level
}) {
    const vocabularyChartContainerQckAnltcs = document.getElementById("vocabularyChartContainerQckAnltcs");
    vocabularyChartContainerQckAnltcs.innerHTML = `<img class="vocabulary-chart-quick-analytics" src="/tmp/vocabulary_chart_small.png">`;

    const recurringWordsContainerQckAnltcs = document.getElementById("recurringWordsContainerQckAnltcs");
    recurringWordsContainerQckAnltcs.innerHTML = recurringLemmasTable;

    const frequentPhrasesContainerQckAnltcs = document.getElementById("frequentPhrasesContainerQckAnltcs");
    frequentPhrasesContainerQckAnltcs.innerHTML = trigramsTable;

    const levelSpan = document.getElementById("levelSpan");
    levelSpan.innerHTML = level;

    // Gradient depending on CEFR level
    if (level === "A1" || level === "A2") {
        levelSpan.style.backgroundImage = "linear-gradient(to right top, #FFCF32, #FFE89C)";
    } else if (level === "B1" || level === "B2") {
        levelSpan.style.backgroundImage = "linear-gradient(to right top, #0f60fd, #8fb9fd)";
    } else if (level === "C1" || level === "C2") {
        levelSpan.style.backgroundImage = "linear-gradient(to right top, #6B66FF, #9C99FF)";
    }

    const statsInputContainer = document.getElementById("statsInputContainer");
    statsInputContainer.innerHTML = statsTable;

    // Appearance
    setTimeout(() => vocabularyChartContainerQckAnltcs.style.maxHeight = `300px`, 1150);

    const quickAnalytics = document.getElementById("quickAnalytics");
    setTimeout(() => quickAnalytics.style.display = `block`, 550);
    setTimeout(() => quickAnalytics.style.animation = `slide1 0.55s ease forwards`, 550);
    setTimeout(() => quickAnalytics.style.animation = `none`, 1150);

    const quickAnalyticsElements = quickAnalytics.getElementsByTagName("li");
    for (let i = 0; i < quickAnalyticsElements.length; i++) {
        quickAnalyticsElements[i].style.animation = `slide2 0.${5 + i}5s ease forwards`;
    }

    const levelAndStatsContainer = document.getElementById("levelAndStatsContainer");
    setTimeout(() => levelAndStatsContainer.style.maxHeight = `80px`, 1600);
};

function addMeasurementsToAnltcs({
    trigramsTable,
    statsTable,
    academicFormulasTable,
    academicCollocationsTable,
    academicWordsTable,
    recurringLemmasTable,
    level,
    recurringLemmas,
    lenAcademicFormulas,
    lenAcademicCollocations,
    lenAcademicWords
}) {
    const vocabularyChartAnalytics = document.getElementById("vocabularyChartAnalytics");
    vocabularyChartAnalytics.src = "/tmp/vocabulary_chart.png";

    const recurringLemmasTableContainerAnltcs = document.getElementById("recurringLemmasTableContainerAnltcs");
    recurringLemmasTableContainerAnltcs.innerHTML = recurringLemmasTable;

    const recurringLemma1 = document.getElementById("recurringLemma1");
    recurringLemma1.innerHTML = recurringLemmas[0];
    const recurringLemma2 = document.getElementById("recurringLemma2");
    recurringLemma2.innerHTML = recurringLemmas[1];
    const recurringLemma3 = document.getElementById("recurringLemma3");
    recurringLemma3.innerHTML = recurringLemmas[2];

    if (lenAcademicFormulas > 1 || lenAcademicWords > 1 || lenAcademicCollocations > 1) {
        const academicFormulasTablesContainerAnltcs = document.getElementById("academicFormulasTablesContainerAnltcs");
        academicFormulasTablesContainerAnltcs.innerHTML += academicFormulasTable;
        academicFormulasTablesContainerAnltcs.innerHTML += academicCollocationsTable;
        academicFormulasTablesContainerAnltcs.innerHTML += academicWordsTable;
    } else {
        const academicFormulasContainerAnltcs = document.getElementById("academicFormulasContainerAnltcs");
        academicFormulasContainerAnltcs.remove();

        const trigramsContainerAnltcs = document.getElementById("trigramsContainerAnltcs");
        trigramsContainerAnltcs.style.flexDirection = "row-reverse";
    }

    const trigramsTableContainerAnltcs = document.getElementById("trigramsTableContainerAnltcs");
    trigramsTableContainerAnltcs.innerHTML = trigramsTable;
};

function tokenClassFromLevel(level) {
    const map = {
        A1: "level0",
        A2: "level1",
        B1: "level2",
        B2: "level3",
        C1: "level4",
        C2: "level5",
    };
    return map[level] || "no_data";
};

function changeLevelHighlighting(buttonID, level, borderColor, backColor) {
    const levelButton = document.getElementById(buttonID);
    const levelSpans = document.getElementsByClassName(level);
    if (levelButton.checked === true) {
        for (let i = 0; i < levelSpans.length; i++) {
            levelSpans[i].style.border = "1px solid " + borderColor;
            levelSpans[i].style.backgroundColor = backColor;
        }
    } else {
        for (let i = 0; i < levelSpans.length; i++) {
            levelSpans[i].style.border = "1px solid #DADADA";
            levelSpans[i].style.backgroundColor = "#f9f9f982";
        }
    }
}

function checkUncheck() {
    let checked_value;
    const button = document.getElementById("checkUncheckButtonInstruments");
    const instrumentsContainer = document.getElementById(`instrumentsContainer`);
    const instrumentsCheckboxes = instrumentsContainer.getElementsByTagName("input");
    if (button.innerText === "Uncheck all") {
        checked_value = false;
        button.innerText = "Check all";
    } else {
        checked_value = true;
        button.innerText = "Uncheck all";
    }
    for (let i = 0; i < instrumentsCheckboxes.length; i++) {
        console.log(instrumentsCheckboxes[i].checked);
        console.log(checked_value);
        if (checked_value !== instrumentsCheckboxes[i].checked) {
            instrumentsCheckboxes[i].click();
        }
    }
}

function copyEssay() {
  const innerText = document.getElementById("markedUpEssayArea").innerText;
  const regexPunct = /(\s)(?<punct>[^-\w\s]+)/gi;
  let text = innerText.replace(regexPunct, "$<punct>");
  const regexDash = /(\s)([-]+)(\s)/gi;
  text = text.replace(regexDash, "-");
  navigator.clipboard.writeText(text).then(() => {
    createToast(`essayCopied`);
  },() => {
    createToast(`essayNotCopied`);
  });
}

function startOver() {
  if (confirm("Do you want to clear the essay and the stats and start over?")) {
    location.reload();
  }
}

async function getReplacements(tokenID, posTag) {
    const url = `${GET_REPLACEMENTS_ENDPOINT}`;
    const tokenText = document.getElementById(tokenID).textContent;
    const tokenInfo = {
        text: tokenText, 
        level: null,
        pos: posTag
    };

    let replacements;
    try {
        const replacementsResponse = await fetch(url, 
            {
                method: "POST", 
                headers: {"Content-Type": "application/json"}, 
                body: JSON.stringify(tokenInfo)
            });
            replacements = await replacementsResponse.json();
    } catch(error) {
        createToast(`connection`);
        console.log(error);
        return;
    }
    addReplacementsPopUpWindow(tokenID, replacements.lemmas, replacements.levels)
}

function addReplacementsPopUpWindow(tokenID, lemmas, levels) {
    const existingWindow = document.getElementById("replacementsPopUpWindow");
    if (existingWindow) existingWindow.remove();

    const markedUpEssayArea = document.getElementById("markedUpEssayArea");
    const replacementsPopUpWindow = document.createElement("div");
    replacementsPopUpWindow.id = "replacementsPopUpWindow";

    for (let i = 0; i < lemmas.length; i++) {
        const buttonReplacement = document.createElement("button");
        buttonReplacement.id = "buttonReplacement";
        buttonReplacement.classList.add(`button-${tokenClassFromLevel(levels[i])}`);
        buttonReplacement.textContent = lemmas[i];
        buttonReplacement.dataset.tokenid = tokenID;
        buttonReplacement.dataset.lemma = lemmas[i];
        buttonReplacement.dataset.level = levels[i];

        replacementsPopUpWindow.appendChild(buttonReplacement);
    }
    if (lemmas.length === 0) {
        const buttonReplacement = document.createElement("button");
        buttonReplacement.id = "noReplacements";
        buttonReplacement.textContent = "no replacements";
        
        replacementsPopUpWindow.innerHTML += buttonReplacement.outerHTML;
    }
    const position = document.getElementById(String(tokenID)).getBoundingClientRect();
    const x_pos = position.left;
    const y_pos = position.top + 30 + window.scrollY;
    replacementsPopUpWindow.style.position = "absolute";
    replacementsPopUpWindow.style.left = x_pos + "px";
    replacementsPopUpWindow.style.top = y_pos + "px";

    replacementsPopUpWindow.addEventListener("mouseleave", () => {
        removeReplacementWindow(3000);
    });

    markedUpEssayArea.innerHTML += replacementsPopUpWindow.outerHTML;
}

function replace(tokenID, lemma, level) {
    const spanObject = document.getElementById(String(tokenID));

    spanObject.textContent = lemma;
    spanObject.classList.remove(`${spanObject.className}`);
    spanObject.classList.add(tokenClassFromLevel(level));

    const replacementsPopUpWindow = document.getElementById("replacementsPopUpWindow");
    replacementsPopUpWindow.addEventListener("mouseleave", () => {
        removeReplacementWindow(1550);
    });
}

function removeReplacementWindow(timeout) {
    const replacementWindow = document.getElementById("replacementsPopUpWindow");
    setTimeout(() => replacementWindow.style.opacity = "0", timeout - 550);
    setTimeout(() => replacementWindow.remove(), timeout);
}

function handleRadioClick(event) {
    const selectedValue = event.target.value;
    const containerIdToShow = selectedValue + 'ContainerQckAnltcs';

    const allDescDivs = document.querySelectorAll('div.desc');

    allDescDivs.forEach(div => {
        if (div.id === containerIdToShow) {
            div.classList.remove('hidden');
        }
        else {
            div.classList.add('hidden');
        }
    });
}

document.body.addEventListener("click", (e) => {
    if (e.target.tagName === "SPAN" && e.target.dataset.pos) {
      getReplacements(e.target.id, e.target.dataset.pos);
    };
    if (e.target.id === "buttonReplacement") {
        const { tokenid, lemma, level } = e.target.dataset;
        replace(tokenid, lemma, level);
    };
  });

document.addEventListener("DOMContentLoaded", () => {
    // Level highlighting checkboxes
    document.getElementById("levelButtonA1").addEventListener("click", () => {
      changeLevelHighlighting("levelButtonA1", "level0", "rgb(255, 232, 156)", "rgba(255, 232, 156, 0.4)");
    });
    document.getElementById("levelButtonA2").addEventListener("click", () => {
      changeLevelHighlighting("levelButtonA2", "level1", "rgb(255, 207, 50)", "rgba(255, 207, 50, 0.45)");
    });
    document.getElementById("levelButtonB1").addEventListener("click", () => {
      changeLevelHighlighting("levelButtonB1", "level2", "rgb(102, 196, 216)", "rgb(102, 196, 216, 0.4)");
    });
    document.getElementById("levelButtonB2").addEventListener("click", () => {
      changeLevelHighlighting("levelButtonB2", "level3", "rgb(82, 130, 242)", "rgba(82, 130, 242, 0.45)");
    });
    document.getElementById("levelButtonC1").addEventListener("click", () => {
      changeLevelHighlighting("levelButtonC1", "level4", "rgb(156, 153, 255)", "rgba(156, 153, 255, 0.4)");
    });
    document.getElementById("levelButtonC2").addEventListener("click", () => {
      changeLevelHighlighting("levelButtonC2", "level5", "rgb(107, 102, 255)", "rgba(107, 102, 255, 0.45)");
    });
  
    // Uncheck button
    document.getElementById("checkUncheckButtonInstruments").addEventListener("click", () => {
      checkUncheck();
    });
  
    // Copy + Start over buttons
    document.getElementById("buttonCopy").addEventListener("click", () => {
      copyEssay();
    });
    document.getElementById("buttonStartOver").addEventListener("click", () => {
      startOver();
    });

    // Radio buttons in the instruments section
    const radioButtons = document.querySelectorAll("input[name='quick-analytics-choice']");
    radioButtons.forEach(radio => {
        radio.addEventListener('click', handleRadioClick);
    });
  });
  