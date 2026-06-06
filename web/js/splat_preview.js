import { app } from "../../../scripts/app.js";

const FOLDER = (() => {
    const m = import.meta.url.match(/\/extensions\/([^/]+)\//);
    return m ? m[1] : "applesharp";
})();

app.registerExtension({
    name: "applesharp.splatpreview",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SharpSplatPreview") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            const container = document.createElement("div");
            container.style.cssText = "width:100%;height:100%;background:#1a1a1a;overflow:hidden;border-radius:4px;";

            const iframe = document.createElement("iframe");
            iframe.style.cssText = "width:100%;height:100%;border:none;";
            iframe.src = `/extensions/${FOLDER}/viewer_splat.html?v=` + Date.now();
            container.appendChild(iframe);

            const widget = this.addDOMWidget("splat_preview", "SPLAT_PREVIEW", container, {
                serialize: false,
                hideOnZoom: false,
                getValue() { return ""; },
                setValue() {},
            });

            widget.computeSize = () => [460, 380];
            this.setSize([460, 430]);

            let iframeReady = false;
            iframe.addEventListener("load", () => { iframeReady = true; });

            const onExecuted = this.onExecuted;
            this.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                const plyFile = message?.ply_file?.[0];
                if (!plyFile) {
                    console.warn("[SHARP Preview] onExecuted: no ply_file in message", message);
                    return;
                }

                const subfolder = message?.subfolder?.[0] ?? "";
                const url = `/applesharp/ply?filename=${encodeURIComponent(plyFile)}&subfolder=${encodeURIComponent(subfolder)}`;

                const send = () => {
                    if (!iframe.contentWindow) return;
                    iframe.contentWindow.postMessage(
                        { type: "LOAD_PLY_URL", url, filename: plyFile },
                        "*"
                    );
                };

                if (iframeReady) send();
                else iframe.addEventListener("load", send, { once: true });
            };

            return r;
        };
    },
});
