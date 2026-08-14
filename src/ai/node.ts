import * as brain from "brain.js";
import type { BrainLib } from "./brain";

(globalThis as { brain?: BrainLib }).brain = brain as unknown as BrainLib;
