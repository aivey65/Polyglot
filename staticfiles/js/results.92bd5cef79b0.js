window.addEventListener("load", () => {
    fetch("/get-accuracy", {
        method: "GET",
    }).then(response => response.json()).then(data => {
        document.getElementById("accuracy-container").innerText = data;
    });
});