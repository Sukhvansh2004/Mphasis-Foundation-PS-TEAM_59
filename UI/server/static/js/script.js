// JavaScript Generate PNR Ranking Rules table
var pnr_ranking_table = [
    ["SSR", "PNR.SSR = SSR"],
    ["No of PAX", "PNR.No Of PAX = Int()"],
    ["Loyality", "PNR.Loyality = Silver"],
    ["Loyality", "PNR.Loyality = Gold"],
    ["Loyality", "PNR.Loyality = Platinum"],
    ["Loyality", "PNR.Loyality = PPlatinum"],
    ["Booking-Type", "PNR.booked_as = Group"],
    ["Paid Service", "PNR.Paid services= Yes"],
    ["Connection", "No. of downline connection = Integer"],
    ["Class", "PNR.class = Economy Class"],
    ["Class", "PNR.class = Business Class"],
    ["Class", "PNR.class = Premium Economy Class"],
    ["Class", "PNR.class = First Class"],
    ["Upgrade Allow", "PNR.allow upgrade = Yes"],
    ["Downgrade Allow", "PNR.allow downgrade = Yes"],
];

var flight_ranking_table = [
    ["Arr_LT_6hrs", "Flight Arrival Delay <= 6hrs"],
    ["Arr_LT_12hrs", "Flight Arrival Delay <= 12hrs"],
    ["Arr_LT_24hrs", "Flight Arrival Delay <= 24hrs"],
    ["Arr_LT_48hrs", "Flight Arrival Delay <= 48hrs"],
    ["Equipment", "Flight Equipment = Int()"],
    ["Same Citypairs", "Source and Destination are same"],
    ["DiffSame Citypairs", "Either Source or Destination are different"],
    ["Different Citypairs", "Both Source and Destination are different"],
    ["SPF_LT_6hrs", "Flight Departure Delay <= 6hrs"],
    ["SPF_LT_12hrs", "Flight Departure Delay <= 12hrs"],
    ["SPF_LT_24hrs", "Flight Departure Delay <= 24hrs"],
    ["SPF_LT_48hrs", "Flight Departure Delay <= 48hrs"],
    ["IsStopover", "No. of Stopovers"],
    ["Category", "Score for Category A Flight"],
    ["Category", "Score for Category B Flight"],
    ["Category", "Score for Category C Flight"],
    ["Category", "Score for Category D Flight"],
]

let pnr_ranking_ele = document.querySelector("#pnr-ranking-table");
for (let i = 0; i < pnr_ranking_table.length; i++) {
    let sno = ("0" + (i + 1)).slice(-2);
    let name = pnr_ranking_table[i][0];
    let condition = pnr_ranking_table[i][1];
    let value = pnr_ranking_score[i];
    let enabled = pnr_ranking_enabled[i] == 1 ? "checked" : "";
    let input_type;
    if (parseInt(value) == value) {
        input_type = "number";
    } else {
        input_type = "text";
    }
    pnr_ranking_ele.innerHTML += `
        <div class="table-row">
            <div class="table-col table-col-1">
                <span>${sno}</span>
            </div>
            <div class="table-col table-col-2">
                <span>${name}</span>
            </div>
            <div class="table-col table-col-3">
                <span>${condition}</span>
            </div>
            <div class="table-col table-col-4">
                <label class="matter-textfield-filled">
                    <input type="${input_type}" class="input_pnr_ranking_score" placeholder=" " value="${value}" rowIndex="${i}"/>
                    <span></span>
                </label>
            </div>
            <div class="table-col table-col-5">
                <label class="matter-checkbox">
                    <input class="input_pnr_ranking_enabled" type="checkbox" ${enabled} rowIndex="${i}"/>
                    <span></span>
                </label>
            </div>
        </div>
        
`;
}

let flight_ranking_ele = document.querySelector("#flight-ranking-table");
for (let i = 0; i < flight_ranking_table.length; i++) {
    let sno = ("0" + (i + 1)).slice(-2);
    let name = flight_ranking_table[i][0];
    let condition = flight_ranking_table[i][1];
    let value = flight_ranking_score[i];
    let enabled = flight_ranking_enabled[i] == 1 ? "checked" : "";
    let input_type;
    if (parseInt(value) == value) {
        input_type = "number";
    } else {
        input_type = "text";
    }
    flight_ranking_ele.innerHTML += `
        <div class="table-row">
            <div class="table-col table-col-1">
                <span>${sno}</span>
            </div>
            <div class="table-col table-col-2">
                <span>${name}</span>
            </div>
            <div class="table-col table-col-3">
                <span>${condition}</span>
            </div>
            <div class="table-col table-col-4">
                <label class="matter-textfield-filled">
                    <input type="${input_type}" class="input_flight_ranking_score" placeholder=" " value="${value}" rowIndex="${i}"/>
                    <span></span>
                </label>
            </div>
            <div class="table-col table-col-5">
                <label class="matter-checkbox">
                    <input class="input_flight_ranking_enabled" type="checkbox" ${enabled} rowIndex="${i}"/>
                    <span></span>
                </label>
            </div>
        </div>
        
`;
}


toastr.options = {
    "closeButton": true,
    "debug": false,
    "newestOnTop": false,
    "progressBar": true,
    "positionClass": "toast-top-right",
    "preventDuplicates": false,
    "onclick": null,
    "showDuration": "300",
    "hideDuration": "1000",
    "timeOut": "5000",
    "extendedTimeOut": "1000",
    "showEasing": "swing",
    "hideEasing": "linear",
    "showMethod": "fadeIn",
    "hideMethod": "fadeOut"
}

const todoValue = document.getElementById("todoText");
const todoAlert = document.getElementById("Alert");
const listItems = document.getElementById("list-items");
const addUpdate = document.getElementById("AddUpdateClick");

// let todo = JSON.parse(localStorage.getItem("todo-list"));
// if (!todo) {
todo = [];
// }

function CreateToDoItems() {
    if (todoValue.value == "") {
        todoAlert.innerText = "Please enter the Flight ID!";
        todoValue.focus();
        setAlertMessage("Please enter the Flight ID!");
        return
    } else {
        let IsPresent = false;
        todo.forEach((element) => {
            if (element.item == todoValue.value) {
                IsPresent = true;
            }
        });

        if (IsPresent) {
            setAlertMessage("This flight already present in the list!");
            return;
        }

        let li = document.createElement("li");
        const todoItems = `<div title="Hit Double Click and Complete">${todoValue.value}</div>
                    <div>
                    <img class="edit todo-controls" onclick="UpdateToDoItems(this)" src="/static/images/pencil-edit.png" height="20em" width="20em"/>
                    <img class="delete todo-controls" onclick="DeleteToDoItems(this)" src="/static/images/bin.png" height="20em" width="20em" /></div>`;
        li.innerHTML = todoItems;
        listItems.appendChild(li);

        if (!todo) {
            todo = [];
        }
        let itemList = { item: todoValue.value, status: false };
        todo.push(itemList);
        setLocalStorage();
    }
    todoValue.value = "";
    setAlertMessage("Flight Added Successfully!");
}


function ReadToDoItems() {
    todo.forEach((element) => {
        let li = document.createElement("li");
        let style = "";
        if (element.status) {
            style = "style='text-decoration: line-through'";
        }
        const todoItems = `<div ${style} title="Hit Double Click and Complete" ">${element.item
            }
    ${style === ""
                ? ""
                : '<img class="todo-controls" src="/static/images/check-mark.png" height="20em" width="20em" />'
            }</div><div>
    ${style === ""
                ? '<img class="edit todo-controls" onclick="UpdateToDoItems(this)" src="/static/images/pencil-edit.png" height="20em" width="20em" />'
                : ""
            }
    <img class="delete todo-controls" onclick="DeleteToDoItems(this)" src="/static/images/bin.png" height="20em" width="20em" /></div>`;
        li.innerHTML = todoItems;
        listItems.appendChild(li);
    });
}
ReadToDoItems();

function UpdateToDoItems(e) {
    if (
        e.parentElement.parentElement.querySelector("div").style.textDecoration ===
        ""
    ) {
        todoValue.value =
            e.parentElement.parentElement.querySelector("div").innerText;
        updateText = e.parentElement.parentElement.querySelector("div");
        addUpdate.setAttribute("onclick", "UpdateOnSelectionItems()");
        addUpdate.setAttribute("src", "/static/images/refresh.png");
        addUpdate.setAttribute("height", "40em");
        addUpdate.setAttribute("width", "40em");
        todoValue.focus();
    }
}

function UpdateOnSelectionItems() {
    let IsPresent = false;
    todo.forEach((element) => {
        if (element.item == todoValue.value) {
            IsPresent = true;
        }
    });

    if (IsPresent) {
        setAlertMessage("This flight already present in the list!");
        return;
    }

    todo.forEach((element) => {
        if (element.item == updateText.innerText.trim()) {
            element.item = todoValue.value;
        }
    });
    setLocalStorage();

    updateText.innerText = todoValue.value;
    addUpdate.setAttribute("onclick", "CreateToDoItems()");
    addUpdate.setAttribute("src", "/static/images/add-sign.png");
    addUpdate.setAttribute("height", "80em");
    addUpdate.setAttribute("width", "80em");
    todoValue.value = "";
    setAlertMessage("Flight ID Updated Successfully!");
}

function DeleteToDoItems(e) {
    let deleteValue =
        e.parentElement.parentElement.querySelector("div").innerText;

    if (confirm(`Are you sure. Due you want to delete this ${deleteValue}!`)) {
        e.parentElement.parentElement.setAttribute("class", "deleted-item");
        todoValue.focus();

        todo.forEach((element) => {
            if (element.item == deleteValue.trim()) {
                todo.splice(element, 1);
            }
        });

        setTimeout(() => {
            e.parentElement.parentElement.remove();
        }, 1000);

        setLocalStorage();
    }
}



function setLocalStorage() {
    localStorage.setItem("todo-list", JSON.stringify(todo));
}

function setAlertMessage(message) {
    todoAlert.removeAttribute("class");
    todoAlert.innerText = message;
    setTimeout(() => {
        todoAlert.classList.add("toggleMe");
    }, 1000);
}



$(document).ready(function () {
    // Add event change event listener to all input elements
    $(".input_pnr_ranking_score").change(function () {
        let rowIndex = this.getAttribute("rowIndex");
        let value = this.value;
        pnr_ranking_score[rowIndex] = value;
        console.log(pnr_ranking_score);
    });

    $(".input_pnr_ranking_enabled").change(function () {
        let rowIndex = this.getAttribute("rowIndex");
        let value = this.checked ? 1 : 0;
        pnr_ranking_enabled[rowIndex] = value;
        console.log(pnr_ranking_enabled);
    });

    // Add event change event listener to all input elements
    $(".input_flight_ranking_score").change(function () {
        let rowIndex = this.getAttribute("rowIndex");
        let value = this.value;
        flight_ranking_score[rowIndex] = value;
        console.log(flight_ranking_score);
    });

    $(".input_flight_ranking_enabled").change(function () {
        let rowIndex = this.getAttribute("rowIndex");
        let value = this.checked ? 1 : 0;
        flight_ranking_enabled[rowIndex] = value;
        console.log(flight_ranking_enabled);
    });


    $("#import-rules-btn-PNR").click(function () {
        let formData = {
            "pnr_ranking_score": pnr_ranking_score,
            "pnr_ranking_enabled": pnr_ranking_enabled
        };
        let xhr = new XMLHttpRequest();
        xhr.open("POST", "/update-pnr-ranking-rules", true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.onload = function () {
            if (xhr.status == 200) {
                let response = JSON.parse(this.responseText);
                console.log(response);
                let status = response["status"];
                let title = response["title"];
                let message = response["message"];
                toastr[status](message, title);
            } else {
                console.log("Error");
                toastr["error"]("Unable to Submit data", "Error");
            }
        }
        xhr.send(JSON.stringify(formData));
    });

    $("#import-rules-btn-Flight").click(function () {
        let formData = {
            "flight_ranking_score": flight_ranking_score,
            "flight_ranking_enabled": flight_ranking_enabled
        };
        let xhr = new XMLHttpRequest();
        xhr.open("POST", "/update-flight-ranking-rules", true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.onload = function () {
            if (xhr.status == 200) {
                let response = JSON.parse(this.responseText);
                console.log(response);
                let status = response["status"];
                let title = response["title"];
                let message = response["message"];
                toastr[status](message, title);
            } else {
                console.log("Error");
                toastr["error"]("Unable to Submit data", "Error");
            }
        }
        xhr.send(JSON.stringify(formData));
    });

    $("#import-dataset-btn").click(function () {
        var formData = new FormData();
        for (let i = 1; i <= 4; i++) {
            formData.append("file-" + i, document.querySelector("#actual-upload-" + i).files[0]);
            formData.append("sheet-" + i, document.querySelector("#dataset-sheet-url-" + i).value);
        }
        var xhr = new XMLHttpRequest();
        xhr.open("POST", "/submit-dataset", true);
        xhr.onload = function () {
            if (xhr.status == 200) {
                let response = JSON.parse(this.responseText);
                console.log(response);
                let status = response["status"];
                let title = response["title"];
                let message = response["message"];
                toastr[status](message, title);
            } else {
                console.log("Error");
                toastr["error"]("Unable to Submit data", "Error");
            }
        }
        xhr.send(formData);
    });

    $("#reschedule-button").click(function () {
        const mode = $("#toggleLabel").text();
        const token = $('#token').val();
        let flights = [];
        
        todo.forEach((element) => {
            flights.push(element.item);
        });
        
        let formData = {
            "Mode": mode,
            "Flights": flights,
            "Token": token
        };
        const outer_div=document.getElementById("outer_div");
        const reschedulePage = outer_div.innerHTML;
        let xhr = new XMLHttpRequest();
        xhr.open("POST", "/reschedule", true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.onload = function () {
            if (xhr.status == 200) {
                let response = JSON.parse(this.responseText);
                console.log('Success:', response);
                let status = response["status"];
                let title = response["title"];
                let message = response["message"];
                toastr[status](message, title);
            } else {
                console.log("Error");
                toastr["error"]("Unable to Reschedule Flights", "Error");
            }
            const outer_div=document.getElementById("outer_div");
            outer_div.innerHTML = reschedulePage;
        };
        xhr.send(JSON.stringify(formData));
        outer_div.innerHTML = `
        <div class="lds-ring">
            <div></div>
            <div></div>
            <div></div>
            <div></div>
        </div>
        `;
        outer_div.style.display = "grid";
        outer_div.style.placeItems = "center";
        outer_div.style.backgroundColor="transparent";
    });
    


    $("#dashboard").click(function () {
        $("#dashboard-content").show();
        $("#inputdataset-content").hide();
        $("#rules-pnr-content").hide();
        $("#rules-flight-content").hide();
        $("#exceptions-content").hide();
        $("#dashboard").addClass("tab-active");
        $("#inputdataset").removeClass("tab-active");
        $("#rules-pnr-content").removeClass("tab-active");
        $("#rules-flight-content").removeClass("tab-active");
        $("#exceptions").removeClass("tab-active");
    });

    $("#inputdataset").click(function () {
        $("#dashboard-content").hide();
        $("#inputdataset-content").show();
        $("#rules-pnr-content").hide();
        $("#rules-flight-content").hide();
        $("#exceptions-content").hide();
        $("#dashboard").removeClass("tab-active");
        $("#inputdataset").addClass("tab-active");
        $("#rules-pnr-content").removeClass("tab-active");
        $("#rules-flight-content").removeClass("tab-active");
        $("#exceptions").removeClass("tab-active");
    });

    $("#rules-pnr").click(function () {
        $("#dashboard-content").hide();
        $("#inputdataset-content").hide();
        $("#rules-pnr-content").show();
        $("#rules-flight-content").hide();
        $("#exceptions-content").hide();
        $("#dashboard").removeClass("tab-active");
        $("#inputdataset").removeClass("tab-active");
        $("#rules-pnr-content").addClass("tab-active");
        $("#rules-flight-content").removeClass("tab-active");
        $("#exceptions").removeClass("tab-active");
    });

    $("#rules-flight").click(function () {
        $("#dashboard-content").hide();
        $("#inputdataset-content").hide();
        $("#rules-pnr-content").hide();
        $("#rules-flight-content").show();
        $("#exceptions-content").hide();
        $("#dashboard").removeClass("tab-active");
        $("#inputdataset").removeClass("tab-active");
        $("#rules-pnr-content").removeClass("tab-active");
        $("#rules-flight-content").addClass("tab-active");
        $("#exceptions").removeClass("tab-active");
    });

    $("#exceptions").click(function () {
        $("#dashboard-content").hide();
        $("#inputdataset-content").hide();
        $("#rules-pnr-content").hide();
        $("#rules-flight-content").hide();
        $("#exceptions-content").show();
        $("#dashboard").removeClass("tab-active");
        $("#inputdataset").removeClass("tab-active");
        $("#rules-pnr-content").removeClass("tab-active");
        $("#rules-flight-content").removeClass("tab-active");
        $("#exceptions").addClass("tab-active");
    });
});

// for (let i = 1; i <= 4; i++) {
//     document.querySelector("#upload-" + i).addEventListener("click", function () {
//         var clickEvent = document.createEvent('MouseEvents');
//         clickEvent.initMouseEvent('click', true, true, window,
//             0, 0, 0, 0, 0, false, false, false, false, 0, null);
//         document.querySelector("#actual-upload-" + i).dispatchEvent(clickEvent);
//     });
//     document.querySelector("#actual-upload-" + i).addEventListener("change", function () {
//         let val = this.value;
//         let filename = val.split(/(\\|\/)/g).pop();
//         document.querySelector("#upload-file-info-" + i).innerHTML = filename;
//     });
// }


var buttons = document.querySelectorAll('.ladda-button');

Array.prototype.slice.call(buttons).forEach(function (button) {

    var resetTimeout;

    button.addEventListener('click', function () {

        if (typeof button.getAttribute('data-loading') === 'string') {
            button.removeAttribute('data-loading');
        }
        else {
            button.setAttribute('data-loading', '');
        }

        clearTimeout(resetTimeout);
        resetTimeout = setTimeout(function () {
            button.removeAttribute('data-loading');
        }, 2000);

    }, false);

});
function updateToggleLabel() {
    const toggle = document.getElementById("modeToggle");
    const label = document.getElementById("toggleLabel");

    if (toggle.checked) {
        label.textContent = "Hybrid";
        label.style.backgroundColor = "white"; // Inverted background color for Hybrid
        label.style.color = "black"; // Inverted text color for Hybrid
    } else {
        label.textContent = "Quantum";
        label.style.backgroundColor = "black"; // Default background color for Quantum
        label.style.color = "white"; // Default text color for Quantum
    }
}


for (let i = 1; i <= 4; i++) {
    document.getElementById(`upload-${i}`).addEventListener('click', function() {
      document.getElementById(`actual-upload-${i}`).click();
    });
  
    document.getElementById(`actual-upload-${i}`).addEventListener('change', function(event) {
      const fileInput = event.target;
      const file = fileInput.files[0];
  
      if (file) {
        document.getElementById(`sheet-link-label-${i}`).style.display = 'none';
        document.getElementById(`upload-${i}`).style.display = 'none';
        document.getElementById(`reupload-${i}`).style.display = 'inline';
        document.getElementById(`or-${i}`).style.display = 'none';
  
        document.getElementById(`upload-file-info-${i}`).textContent = `Uploaded: ${file.name}`;
  
        const fileUrl = URL.createObjectURL(file);
        const csvLink = document.getElementById(`csv-link-${i}`);
        csvLink.href = fileUrl;
        csvLink.style.display = 'inline';
      }
    });
  }