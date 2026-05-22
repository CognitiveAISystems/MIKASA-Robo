Environments & Tasks
====================

.. raw:: html

   <style>
   /* Widen content area for this page only */
   .bd-page-width { max-width: 1560px !important; }
   .bd-content { max-width: 1560px !important; }
   article.bd-article { max-width: 1560px !important; }
   </style>

MIKASA-Robo-VLA contains **90 tasks** across 10 memory types and three
horizon splits.

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Split
     - Tasks
     - Horizon (steps)
     - Typical episode length (at 25 Hz)
   * - **Short**
     - 38
     - 25 – 200
     - 1 – 8 s
   * - **Medium**
     - 30
     - 201 – 600
     - 8 – 24 s
   * - **Long**
     - 22
     - 601 – 2160
     - 24 – 86 s

Memory Types
------------

The table below gives an operational definition of each memory type: what
information must persist across time, why current-frame observations are
insufficient, and what kind of history the agent must use.

.. list-table::
   :header-rows: 1
   :widths: 16 10 42 32

   * - Memory Type
     - # Tasks
     - What must be remembered or tracked
     - Why a reactive policy fails
   * - **Object**
     - 18
     - A previously observed object attribute: colour, shape, identity, or a
       shape-colour binding.
     - The target attribute disappears before the agent must choose.
   * - **Spatial**
     - 14
     - A hidden placement, earlier reference pose, motion context, or spatial
       relation needed for a later manipulation decision.
     - The current scene may omit the hidden target, earlier reference state,
       or trajectory context needed for the spatial action.
   * - **Capacity**
     - 12
     - Multiple unordered items or object attributes, or a count that
       summarises cues exposed together or one at a time.
     - Remembering only one item is insufficient; the agent must preserve the
       full set or its task-relevant aggregate.
   * - **Temporal**
     - 12
     - Information accumulated over time, such as blink counts, elapsed-step
       counts, timed cues, or stage-dependent events.
     - The decision depends on the history of observations, not on a single
       visible cue.
   * - **Negative**
     - 9
     - Which candidates were present before, so the agent can identify the
       missing or odd-one-out candidate later.
     - The correct answer is defined by exclusion rather than by matching a
       visible cue.
   * - **Sequential**
     - 6
     - The order in which multiple targets or cues appeared and must later be
       executed.
     - The same target set can require different actions when its remembered
       order changes.
   * - **Procedural**
     - 6
     - A path-like motor procedure or sequence of movements that must be
       reproduced during execution.
     - The current state does not encode the remaining procedure; the agent
       must recall how the motion should continue.
   * - **Prospective**
     - 5
     - A delayed intention that must remain active while another task is being
       completed.
     - The early goal is no longer salient when the final action is required.
   * - **Tracking**
     - 4
     - A belief about hidden object identity or position while objects move or
       exchange roles.
     - The correct state changes during the episode and must be updated
       continuously.
   * - **Checklist**
     - 4
     - Which required conditions or items have already been tested, verified,
       or completed, and what result each check produced.
     - The current frame cannot recover the completed checks, their outcomes,
       and the remaining required items.

See :doc:`../concepts` for a description of each memory type, and
:doc:`../benchmarking` for the canonical evaluation protocol.

This section is the task catalog.  Each row links to one environment-family
page, where all difficulty settings implemented by the same Python file are
documented together.  For example, :doc:`remember_color` covers
``RememberColor3-VLA-v0``, ``RememberColor5-VLA-v0``,
``RememberColor9-VLA-v0``, and their long-horizon variants:
``RememberColor3-Long-VLA-v0``, ``RememberColor5-Long-VLA-v0``,
``RememberColor9-Long-VLA-v0``.

Use the table for quick task selection, then open a task page for mechanics,
wrapper recommendations, render previews, and collection commands.

Task Overview
-------------

.. raw:: html

   <div style="text-align:center;margin-bottom:6px;font-size:1.1em;color:#666;animation:scroll-hint 1.2s ease-in-out infinite alternate">
     ← scroll →
   </div>
   <style>
     @keyframes scroll-hint {
       from { opacity: 0.25; letter-spacing: 0.05em; }
       to   { opacity: 1;    letter-spacing: 0.25em; }
     }
   </style>

.. list-table::
   :header-rows: 1
   :widths: 11 10 11 10 10 10 9 8 19
   :class: wide-table

   * - Preview
     - Task
     - Episode Length
     - Horizon Split
     - Memory Type
     - Difficulty
     - Long Variant
     - Data Source
     - Language Instruction
   * - .. image:: ../_static/videos/shell_game_touch.gif
          :alt: Shell Game Touch render preview
          :class: task-overview-preview
     - :doc:`ShellGameTouch-VLA-v0 <shell_game_touch>`
     - short 30
     - Short
     - Spatial
     - single setting
     - -
     - PPO
     - Observe which cup hides the ball, wait, then touch that cup.
   * - .. image:: ../_static/videos/shell_game_push.gif
          :alt: Shell Game Push render preview
          :class: task-overview-preview
     - :doc:`ShellGamePush-VLA-v0 <shell_game_push>`
     - short 30
     - Short
     - Spatial
     - single setting
     - -
     - PPO
     - Observe which cup hides the ball, wait, then push that cup forward.
   * - .. image:: ../_static/videos/shell_game_shuffle_touch.gif
          :alt: Shell Game Shuffle Touch render preview
          :class: task-overview-preview
     - :doc:`ShellGameShuffleTouch-VLA-v0 <shell_game_shuffle_touch>`
     - short 60; medium 600
     - Short / Medium
     - Tracking
     - single setting
     - yes (600 steps)
     - PPO / MP
     - Observe which cup hides the ball, track the cups as they shuffle, then touch the correct cup.
   * - .. image:: ../_static/videos/shell_game_color_lamp_touch.gif
          :alt: Shell Game Color Lamp Touch render preview
          :class: task-overview-preview
     - :doc:`ShellGameColorLampTouch-VLA-v0 <shell_game_color_lamp_touch>`
     - short 30
     - Short
     - Spatial
     - single setting
     - -
     - PPO
     - Observe which color is under each cup, then touch the cup matching the lamp color.
   * - .. image:: ../_static/videos/shell_game_shuffle_color_lamp_touch.gif
          :alt: Shell Game Shuffle Color Lamp Touch render preview
          :class: task-overview-preview
     - :doc:`ShellGameShuffleColorLampTouch-VLA-v0 <shell_game_shuffle_color_lamp_touch>`
     - short 60; medium 600
     - Short / Medium
     - Tracking
     - single setting
     - yes (600 steps)
     - PPO / MP
     - Observe which color is under each cup, track the cups as they shuffle, then touch the cup matching the lamp color.
   * - .. image:: ../_static/videos/intercept.gif
          :alt: Intercept render preview
          :class: task-overview-preview
     - :doc:`InterceptSlow/Medium/Fast-VLA-v0 <intercept>`
     - short 60
     - Short
     - Spatial
     - slow / medium / fast
     - -
     - PPO
     - Intercept the rolling ball by moving to its path and deflecting it toward the target.
   * - .. image:: ../_static/videos/intercept_grab.gif
          :alt: Intercept Grab render preview
          :class: task-overview-preview
     - :doc:`InterceptGrabSlow/Medium/Fast-VLA-v0 <intercept_grab>`
     - short 60
     - Short
     - Spatial
     - slow / medium / fast
     - -
     - PPO
     - Intercept the rolling ball and grasp it to stop it.
   * - .. image:: ../_static/videos/rotate_lenient.gif
          :alt: Rotate Lenient render preview
          :class: task-overview-preview
     - :doc:`RotateLenientPos/PosNeg-VLA-v0 <rotate_lenient>`
     - short 60
     - Short
     - Spatial
     - positive / signed angles
     - -
     - PPO
     - Rotate the peg by {angle_deg} degrees to match the target angle.
   * - .. image:: ../_static/videos/rotate_strict.gif
          :alt: Rotate Strict render preview
          :class: task-overview-preview
     - :doc:`RotateStrictPos/PosNeg-VLA-v0 <rotate_strict>`
     - short 90
     - Short
     - Spatial
     - positive / signed angles
     - -
     - PPO
     - Rotate the peg by {angle_deg} degrees to match the target angle while keeping the center of the peg in place.
   * - .. image:: ../_static/videos/take_it_back.gif
          :alt: Take It Back render preview
          :class: task-overview-preview
     - :doc:`TakeItBack-VLA-v0 <take_it_back>`
     - short 60
     - Short
     - Spatial
     - single setting
     - -
     - PPO
     - Push the cube onto the red target, and when the target changes color, return the cube to its original position.
   * - .. image:: ../_static/videos/remember_color.gif
          :alt: Remember Color render preview
          :class: task-overview-preview
     - :doc:`RememberColor3/5/9-VLA-v0 <remember_color>`
     - short 25; medium 600
     - Short / Medium
     - Object
     - 3 / 5 / 9
     - yes (600 steps)
     - PPO / MP
     - Observe the cube's color, wait, then touch the cube of the same color.
   * - .. image:: ../_static/videos/remember_shape.gif
          :alt: Remember Shape render preview
          :class: task-overview-preview
     - :doc:`RememberShape3/5/9-VLA-v0 <remember_shape>`
     - short 25; medium 600
     - Short / Medium
     - Object
     - 3 / 5 / 9
     - yes (600 steps)
     - PPO / MP
     - Observe the object's shape, wait, then touch the object of the same shape.
   * - .. image:: ../_static/videos/remember_shape_and_color.gif
          :alt: Remember Shape And Color render preview
          :class: task-overview-preview
     - :doc:`RememberShapeAndColor3x2/3x3/5x3-VLA-v0 <remember_shape_and_color>`
     - short 25; medium 600
     - Short / Medium
     - Object
     - 3x2 / 3x3 / 5x3
     - yes (600 steps)
     - PPO / MP
     - Observe the object's shape and color, wait, then touch the object of the same shape and color.
   * - .. image:: ../_static/videos/find_imposter_color.gif
          :alt: Find Imposter Color render preview
          :class: task-overview-preview
     - :doc:`FindImposterColor3/5/9-VLA-v0 <find_imposter_color>`
     - short 25
     - Short
     - Negative
     - 3 / 5 / 9
     - -
     - PPO
     - Observe the cubes shown, wait, then touch the cube whose color was not present before.
   * - .. image:: ../_static/videos/find_imposter_shape.gif
          :alt: Find Imposter Shape render preview
          :class: task-overview-preview
     - :doc:`FindImposterShape3/5/9-VLA-v0 <find_imposter_shape>`
     - short 25
     - Short
     - Negative
     - 3 / 5 / 9
     - -
     - PPO
     - Observe the shapes shown, wait, then touch the object whose shape was not present before.
   * - .. image:: ../_static/videos/find_imposter_shape_and_color.gif
          :alt: Find Imposter Shape And Color render preview
          :class: task-overview-preview
     - :doc:`FindImposterShapeAndColor3x2/3x3/5x3-VLA-v0 <find_imposter_shape_and_color>`
     - short 25
     - Short
     - Negative
     - 3x2 / 3x3 / 5x3
     - -
     - PPO
     - Observe the objects shown, wait, then touch the object whose shape and color combination was not present before.
   * - .. image:: ../_static/videos/bunch_of_colors.gif
          :alt: Bunch Of Colors render preview
          :class: task-overview-preview
     - :doc:`BunchOfColors3/5/7-VLA-v0 <bunch_of_colors>`
     - medium 400; long 700
     - Medium / Long
     - Capacity
     - 3 / 5 / 7
     - yes (700 steps)
     - MP
     - Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button.
   * - .. image:: ../_static/videos/seq_of_colors.gif
          :alt: Seq Of Colors render preview
          :class: task-overview-preview
     - :doc:`SeqOfColors3/5/7-VLA-v0 <seq_of_colors>`
     - medium 400; long 800; long 1000; long 1200
     - Medium / Long
     - Capacity
     - 3 / 5 / 7
     - yes (800 / 1000 / 1200 steps)
     - MP
     - Observe which colored cubes appear during the cue, wait, then touch all of them in any order and press the center button.
   * - .. image:: ../_static/videos/chain_of_colors.gif
          :alt: Chain Of Colors render preview
          :class: task-overview-preview
     - :doc:`ChainOfColors3/5/7-VLA-v0 <chain_of_colors>`
     - medium 400; long 800; long 1000; long 1200
     - Medium / Long
     - Sequential
     - 3 / 5 / 7
     - yes (800 / 1000 / 1200 steps)
     - MP
     - Observe which colored cubes appear during the cue, wait, then touch all of them in the same order as the cubes were shown and press the center button.
   * - .. image:: ../_static/videos/trace_shape.gif
          :alt: Trace Shape render preview
          :class: task-overview-preview
     - :doc:`TraceShapeEasy/Medium/Hard-VLA-v0 <trace_shape>`
     - medium 250; medium 300; medium 350
     - Medium
     - Procedural
     - easy / medium / hard
     - -
     - MP
     - Watch the red cube trace a shape on the table. When the lamp turns green, pick up the green cube and trace exactly the same shape.
   * - .. image:: ../_static/videos/trace_shape_seq.gif
          :alt: Trace Shape Seq render preview
          :class: task-overview-preview
     - :doc:`TraceShapeSeqEasy/Medium/Hard-VLA-v0 <trace_shape_seq>`
     - long 1500
     - Long
     - Procedural
     - easy / medium / hard
     - yes (1500 steps)
     - MP
     - Watch the red cube trace a sequence of shapes. When the lamp turns green, pick up the green cube and trace the same sequence in order. After finishing all shapes, press the button to submit your answer.
   * - .. image:: ../_static/videos/blink_count_button_press.gif
          :alt: Blink Count Button Press render preview
          :class: task-overview-preview
     - :doc:`BlinkCountButtonPressEasy/Medium/Hard-VLA-v0 <blink_count_button_press>`
     - short 150; short 200; medium 300; long 1200
     - Short / Medium / Long
     - Temporal
     - easy / medium / hard
     - yes (1200 steps)
     - MP
     - Count how many times the blue lamp blinks, press the red button exactly that many times when the red lamp turns green, then press the black button to submit your answer.
   * - .. image:: ../_static/videos/timed_transfer.gif
          :alt: Timed Transfer render preview
          :class: task-overview-preview
     - :doc:`TimedTransferEasy/Medium/Hard-VLA-v0 <timed_transfer>`
     - short 200; medium 250; medium 300; medium 600; long 900; long 1200
     - Short / Medium / Long
     - Temporal
     - easy / medium / hard
     - yes (600 / 900 / 1200 steps)
     - MP
     - When the white lamp turns green, start counting steps from that exact moment. Move the blue cube from the green disc to the red disc exactly on step 100 of that count.
   * - .. image:: ../_static/videos/batteries_checker_easy.gif
          :alt: Batteries Checker Easy render preview
          :class: task-overview-preview
     - :doc:`BatteriesCheckerEasy-3/6-VLA-v0 <batteries_checker_easy>`
     - medium 540; long 1080
     - Medium / Long
     - Checklist
     - 3 / 6 batteries
     - yes (1080 steps)
     - MP
     - Find all working batteries by inserting each one into the socket, observing the lamp result, and then pressing the button to confirm.
   * - .. image:: ../_static/videos/batteries_checker_hard.gif
          :alt: Batteries Checker Hard render preview
          :class: task-overview-preview
     - :doc:`BatteriesCheckerHard-3/6-VLA-v0 <batteries_checker_hard>`
     - long 1080; long 2160
     - Long
     - Checklist
     - 3 / 6 batteries
     - yes (1080 / 2160 steps)
     - MP
     - Find all working batteries by inserting each one into the socket, observing the lamp result, returning it from the socket to its initial slot, and then pressing the button to confirm.
   * - .. image:: ../_static/videos/gather_and_recall.gif
          :alt: Gather And Recall render preview
          :class: task-overview-preview
     - :doc:`GatherAndRecall1/3/5/7/9-VLA-v0 <gather_and_recall>`
     - short 200; medium 400; medium 600; long 800; long 1000
     - Short / Medium / Long
     - Prospective
     - 1 / 3 / 5 / 7 / 9
     - yes (800 / 1000 steps)
     - MP
     - Move all cubes onto the disc. A lamp will briefly flash while you work. After all cubes are placed, press the button matching the flash color.

.. toctree::
   :hidden:
   :maxdepth: 1

   shell_game_touch
   shell_game_push
   shell_game_shuffle_touch
   shell_game_color_lamp_touch
   shell_game_shuffle_color_lamp_touch
   intercept
   intercept_grab
   rotate_lenient
   rotate_strict
   take_it_back
   remember_color
   remember_shape
   remember_shape_and_color
   find_imposter_color
   find_imposter_shape
   find_imposter_shape_and_color
   bunch_of_colors
   seq_of_colors
   chain_of_colors
   trace_shape
   trace_shape_seq
   blink_count_button_press
   timed_transfer
   batteries_checker_easy
   batteries_checker_hard
   gather_and_recall
