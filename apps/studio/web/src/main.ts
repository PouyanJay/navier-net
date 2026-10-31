import { mount } from "svelte";

import App from "./App.svelte";
import "./styles/base.css";

const target = document.getElementById("app");
if (!target) throw new Error("index.html has no #app element");

export default mount(App, { target });
