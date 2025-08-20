export const toastDetails = {
    notEnoughWords: {
        text: "Please, enter more words (250+)"
    },
    tokensNotRecognized: {
        text: "TAVR has not recognized enough English words for analysis"
    },
    noConnectionWithServer: {
        text: "No connection with the TAVR server"
    },
    essayCopied: {
      text: "The essay with the changes made has been copied"
    },
    essayNotCopied: {
      text: "Failed to copy :("
    }
}

export const list = document.getElementById("notificationsList");
export function createToast (className) {
    const toast = document.createElement("li");
    toast.className = `toast-${className}`;
    const text = toastDetails[className].text;
    toast.innerHTML = `<span class="${className}">${text}</span>`;
    list.appendChild(toast);
    removeToast(toast);
}

export function removeToast(toast) {
    setTimeout(() => toast.style.animation = "fadeOut 1.5s ease-out", 3500)
    setTimeout(() => toast.remove(), 5000);
}