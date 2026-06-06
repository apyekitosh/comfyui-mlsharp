import { app } from "../../../scripts/app.js";

const FOLDER = (() => {
    const m = import.meta.url.match(/\/extensions\/([^/]+)\//);
    return m ? m[1] : "comfyui-mlsharp";
})();

const STATE_WIDGET = "interactive_state";

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name) || null;
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) return;
    widget.value = value;
    widget.callback?.(value);
}

function getWidgetValue(node, name, fallback) {
    const w = getWidget(node, name);
    return w ? w.value : fallback;
}

function markNodeChanged(node) {
    node.setDirtyCanvas?.(true, true);
    node.graph?.afterChange?.();
    app.graph?.afterChange?.();
    app.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "mlsharp.shotrender",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SharpShotRender") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);

            // Hide the interactive_state string widget
            const stateWidget = getWidget(this, STATE_WIDGET);
            if (stateWidget) stateWidget.hidden = true;

            const container = document.createElement("div");
            container.style.cssText = "width:100%;height:100%;min-height:360px;display:flex;flex-direction:column;overflow:hidden;background:#1a1a1a;border-radius:4px;";

            const iframe = document.createElement("iframe");
            iframe.style.cssText = "width:100%;height:100%;border:none;";
            iframe.src = `/extensions/${FOLDER}/viewer_shot.html?v=` + Date.now();
            container.appendChild(iframe);

            this.addDOMWidget("sharp_shot_viewer", "SHARP_SHOT_VIEWER", container, {
                serialize: false,
                hideOnZoom: false,
                getValue() { return ""; },
                setValue() {},
            });

            this.setSize([520, 580]);

            // ── onExecuted: load PLY into viewer ──

            const onExecuted = this.onExecuted;
            this.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                const plyFile = message?.ply_file?.[0];
                if (!plyFile) return;

                const subfolder = message?.subfolder?.[0] ?? "";
                const url = `/mlsharp/ply?filename=${encodeURIComponent(plyFile)}&subfolder=${encodeURIComponent(subfolder)}`;

                const sendAll = () => {
                    if (!iframe.contentWindow) return;
                    iframe.contentWindow.postMessage(
                        { type: "LOAD_PLY_URL", url, filename: plyFile },
                        "*"
                    );
                    const st = getWidgetValue(this, STATE_WIDGET, "");
                    if (st) {
                        try {
                            const parsed = JSON.parse(st);
                            if (Math.abs(Number(parsed.distance) || 0) > 1e-5) {
                                setTimeout(() => {
                                    iframe.contentWindow?.postMessage(
                                        { type: "REMOTE_STATE", state: parsed },
                                        "*"
                                    );
                                }, 500);
                            }
                        } catch (_) { /* ignore */ }
                    }
                };

                if (iframe.contentDocument?.readyState === "complete") {
                    sendAll();
                } else {
                    iframe.addEventListener("load", () => sendAll(), { once: true });
                }
            };

            return r;
        };
    },

    async setup() {
        window.addEventListener("message", (event) => {
            const msg = event.data;
            if (!msg || msg.type !== "SHARP_CAMERA_CHANGED" || !msg.state) return;

            // Find the SharpShotRender node whose iframe sent this message
            // Walk all nodes and match by iframe contentWindow
            for (const node of app.graph?._nodes ?? []) {
                if (node.type !== "SharpShotRender") continue;
                const iframe = node.widgets?.find(w => w.name === "sharp_shot_viewer")?.element?.querySelector?.("iframe");
                if (!iframe || iframe.contentWindow !== event.source) continue;

                setWidgetValue(node, STATE_WIDGET, JSON.stringify(msg.state));
                markNodeChanged(node);
                return;
            }
        });
    },
});
