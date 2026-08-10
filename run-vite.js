globalThis.CustomEvent = class CustomEvent extends Event {
  constructor(event, params = {}) {
    super(event, params);
    this.detail = params.detail || null;
  }
};
import('./node_modules/vite/bin/vite.js');
