const updateGradioImage = async (element, url, name) => 
{
    const waitForElementToBeInDocument = (parent, selector) =>
        Promise.race([stripper_waitForElement(parent, selector, true), stripper_timeout(10000)]);

    const waitForElementToBeRemoved = (parent, selector) =>
        Promise.race([stripper_waitForElement(parent, selector, false), stripper_timeout(10000)]);

    const blob = await (await fetch(url)).blob();
    const file = new File([blob], name, { type: "image/png" });
    const dt = new DataTransfer();
    dt.items.add(file);

    function getClearButton() {
        let clearButton = null;
        let clearLabel = null;

        let allButtons = element.querySelectorAll("button");
        if (allButtons.length > 0) {
            for (let button of allButtons) {
                let label = button.getAttribute("aria-label");
                if (label && !label.includes("Edit") && !label.includes("Éditer")) {
                    clearButton = button;
                    clearLabel = label;
                    break;
                }
            }
        }
        return [clearButton, clearLabel];
    }

    const [clearButton, clearLabel] = getClearButton();

    if (clearButton) {
        clearButton?.click();
        await waitForElementToBeRemoved(element, `button[aria-label='${clearLabel}']`);
    }

    const input = element.querySelector("input[type='file']");
    input.value = "";
    input.files = dt.files;
    input.dispatchEvent(
        new Event("change", {
            bubbles: true,
            composed: true,
        })
    );
    await waitForElementToBeInDocument(element, "button");
};
const stripper_waitForElement = async (parent, selector, exist) => {
    return new Promise((resolve) => {
        const observer = new MutationObserver(() => {
            if (!!parent.querySelector(selector) != exist) {
                return;
            }
            observer.disconnect();
            resolve(undefined);
        });

        observer.observe(parent, {
            childList: true,
            subtree: true,
        });

        if (!!parent.querySelector(selector) == exist) {
            resolve(undefined);
        }
    });
};

const stripper_timeout = (ms) => {
    return new Promise(function (resolve, reject) {
        setTimeout(() => reject("Timeout"), ms);
    });
};

async function stripper_clearSelMask() {
    const waitForElementToBeInDocument = (parent, selector) =>
        Promise.race([stripper_waitForElement(parent, selector, true), stripper_timeout(1000)]);

    const elemId = "#stripper_sel_mask";

    const targetElement = document.querySelector(elemId);
    if (!targetElement) {
        return;
    }
    await waitForElementToBeInDocument(targetElement, "button[aria-label='Clear']");

    targetElement.style.transform = null;
    targetElement.style.zIndex = null;
    targetElement.style.overflow = "auto";

    const selMaskClear = targetElement.querySelector("button[aria-label='Clear']");
    if (!selMaskClear) {
        return;
    }
    const removeImageButton = targetElement.querySelector("button[aria-label='Remove Image']");
    if (!removeImageButton) {
        return;
    }
    selMaskClear?.click();

    if (typeof stripper_clearSelMask.clickRemoveImage === "undefined") {
        stripper_clearSelMask.clickRemoveImage = () => {
            targetElement.style.transform = null;
            targetElement.style.zIndex = null;
        };
    } else {
        removeImageButton.removeEventListener("click", stripper_clearSelMask.clickRemoveImage);
    }
    removeImageButton.addEventListener("click", stripper_clearSelMask.clickRemoveImage);
}