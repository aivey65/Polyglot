window.addEventListener("load", () => {
    fetch("/get-accuracy", {
        method: "GET",
    }).then(response => response.json()).then(data => {
        console.log(data)
        document.getElementById("accuracy-container").innerText = data;
    });
});