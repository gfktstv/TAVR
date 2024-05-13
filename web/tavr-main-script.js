// Character counter
document.getElementById('inputData').addEventListener('input', function() {
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

// Send an essay to TAVR and get measurements
function getMeasurementsFromTAVR(inputData) {
  // Create a new XMLHttpRequest object
  const xhr = new XMLHttpRequest();
  // Define the endpoint URL of the Python server
  const url = 'http://localhost:5000/receive_data';
  // Set up the request
  xhr.open('POST', url, true);
  xhr.setRequestHeader('Content-Type', 'application/json');
  // Define a callback function to handle the response from the server
  xhr.onreadystatechange = function () {
      if (xhr.readyState === XMLHttpRequest.DONE) {
          if (xhr.status === 200) {
              console.log('Data successfully sent to Python server.');
              // Optionally handle the response from the server
              console.log(xhr.responseText);
          } else {
              console.error('Failed to send data to Python server.');
          }
      }
  };
  // Convert JavaScript object to JSON string
  const jsonData = JSON.stringify({ data: inputData });
  // Send the request with the JSON data
  xhr.send(jsonData);
  // Get measurements from TAVR
  measurements = xhr.responseText
  return measurements
}

// Process essay and show assessment
function showAssessment() {
  // Sends essay to TAVR
  const inputData = document.getElementById('inputData')
  measurements = getMeasurementsFromTAVR(inputData.value)
  console.log(measurements)
  // Removes the buttonAssess and adds a buttonClear and a buttonCopy
  const buttonAssess = document.getElementById('buttonAssess');
  buttonAssess.remove();
  const characterCount = document.getElementById('characterCount');
  characterCount.remove();
  const inputButton = document.getElementById('inputButton');
  inputButton.innerHTML += `
    <button id="buttonCopy" onclick="copyText()"><strong>Copy the essay</string></button>
    <button id="buttonClear" onclick="clearText()"><strong>Clear the essay</strong></button>
  `;
}

// Change a view of the text after processing it
function changeText() {

}

// Buttons "clear" and "copy"

function clearText() {
  if (confirm("Do you want to clear the essay and the stats?")) {
    location.reload()
  }
}

function copyText() {
  const inputData = document.getElementById('inputData').value
  navigator.clipboard.writeText(inputData).then(() => {
    console.log('Content copied to clipboard');
    /* Resolved - text copied to clipboard successfully */
  },() => {
    console.error('Failed to copy');
    /* Rejected - text failed to copy to the clipboard */
  });
}