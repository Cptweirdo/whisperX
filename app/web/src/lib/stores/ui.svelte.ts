// Small cross-screen UI intents. `newRecording` lets the sidebar (on any page)
// ask the dashboard to open the upload dialog; the New Recording button on other
// pages navigates home first (mirrors base.html's newRecording()).
import { router } from "../router.svelte";

class UiStore {
  newRecordingRequested = $state(false);

  requestNewRecording() {
    this.newRecordingRequested = true;
    if (router.current.name !== "dashboard") router.navigate("/");
  }

  consumeNewRecording(): boolean {
    if (this.newRecordingRequested) {
      this.newRecordingRequested = false;
      return true;
    }
    return false;
  }
}

export const ui = new UiStore();
