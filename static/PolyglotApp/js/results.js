window.addEventListener("load", () => {
    fetch("/get-accuracy", {
        method: "GET",
    }).then(response => response.json()).then(data => {
        document.getElementById("accuracy-container").innerText = (parseFloat(data.toFixed(2)) * 100) + "%";
    });
});