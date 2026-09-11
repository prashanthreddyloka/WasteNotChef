import { useReducedMotion } from "framer-motion";
import type { PointerEvent } from "react";

export function KitchenScene() {
  const reducedMotion = useReducedMotion();
  function tilt(event: PointerEvent<HTMLDivElement>) {
    if (reducedMotion || event.pointerType !== "mouse") return;
    const box = event.currentTarget.getBoundingClientRect();
    event.currentTarget.style.setProperty("--scene-x", `${(0.5 - (event.clientY - box.top) / box.height) * 8}deg`);
    event.currentTarget.style.setProperty("--scene-y", `${((event.clientX - box.left) / box.width - 0.5) * 10}deg`);
  }
  return (
    <div className="kitchen-scene" onPointerMove={tilt} onPointerLeave={event => {
      event.currentTarget.style.setProperty("--scene-x", "0deg");
      event.currentTarget.style.setProperty("--scene-y", "0deg");
    }} aria-label="A dimensional bowl of fresh vegetables, with a grocery receipt and meal-planning note" role="img">
      <div className="scene-orbit orbit-one" /><div className="scene-orbit orbit-two" />
      <div className="scene-stage" aria-hidden="true">
        <div className="scene-shadow" />
        <div className="scene-board"><span>THE EVERYDAY KITCHEN</span></div>
        <div className="scene-receipt"><span>GOOD FOOD.</span><b>Less waste.</b><i /><i /><i /><div>GROCERIES → POSSIBILITIES</div></div>
        <div className="scene-bowl"><div className="bowl-rim" /><div className="bowl-body" /></div>
        <div className="vegetable leaf leaf-one" /><div className="vegetable leaf leaf-two" /><div className="vegetable leaf leaf-three" />
        <div className="vegetable tomato tomato-one"><span>✦</span></div>
        <div className="vegetable tomato tomato-two"><span>✦</span></div>
        <div className="vegetable avocado"><div /></div>
        <div className="vegetable lemon" />
        <div className="scene-note"><span className="note-dot" />A little planning.<br /><strong>A lot less waste.</strong></div>
        <div className="scene-spark spark-one">✧</div><div className="scene-spark spark-two">✧</div>
      </div>
      <span className="scene-caption">FRESH IDEAS START WITH WHAT YOU HAVE</span>
    </div>
  );
}
