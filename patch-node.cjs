const util = require('node:util');
if (!util.styleText) {
  util.styleText = (format, text) => text;
}
if (typeof globalThis.CustomEvent === 'undefined') {
  globalThis.CustomEvent = class CustomEvent extends Event {
    constructor(event, params = {}) {
      super(event, params);
      this.detail = params.detail || null;
    }
  };
}
