# Can we give a language model a live sense of time?

**We built a working prototype that helps an AI assistant keep track of passing time and changes while it is busy.** It runs locally and does not require retraining either model.

The useful result is straightforward: a small program watches the clock and incoming information, then gives the language model an up-to-date account of what is happening. If the situation changes while the model is writing, the program can stop that old answer from reaching the user.

The model is using a clock supplied to it. It has not learned to keep time independently.

## What we were trying to achieve

An assistant should be able to notice things such as:

- “I have been waiting for five seconds.”
- “That deadline has already passed.”
- “My last measurement is now too old to rely on.”
- “An alarm arrived while I was answering, so my answer needs to change.”

We also wanted to explore whether TimesFM—a model that predicts how measurements change over time—could help the assistant decide when to check again.

These are different jobs. An ordinary clock can measure elapsed time exactly. A forecasting model is useful only if its predictions help the assistant anticipate a change.

## What happened in the final live test

The test ran for **24.69 seconds** on an Apple M4 Max Mac. The program received **201 measurements** and called the language model **7 times**.

First, we let real time pass. When asked how long the run had been active, the model answered **4.528 seconds**. The clock value supplied to it was about **4.5277 seconds**. It also correctly reported that a deadline had expired. When given an intentionally old copy of the situation instead, it said the deadline was still pending.

Next, we started a longer answer. After the model had begun producing text, we introduced an alarm and set an urgent deadline. The program received the alarm and handled the deadline while the model continued writing. Once that answer finished, the program rejected it because it had been written from the old situation. The next answer correctly reported the alarm and expired deadline.

We then paused the incoming measurements. The model correctly identified the last measurement as too old under the test's one-second freshness rule. We also checked that an answer could expire simply because time passed, even without a new event.

Finally, the program collected enough recent measurements to call TimesFM. TimesFM produced a forecast, and the program used it to schedule another check. That check actually ran.

The main demonstration used simulated measurements produced as real time passed. The data did not come from a physical sensor. However, the waiting, model generation, alarm arrival, deadline handling, and rejection of the old answer all happened in real time. A separate test used actual events appended to a file while the model was generating.

## What passed

| Test | Result | What it establishes |
|---|---|---|
| Clock and event handling, without TimesFM | 10 of 10 scripted checks passed | The system can use a clock, track changes, reject an old answer, and refresh its response. |
| Clock and event handling, with TimesFM | 12 of 12 scripted checks passed | The forecasting model can run on newly collected data and schedule a later check. |
| Events written to a real file during generation | Passed | The public watcher can receive an event and deadline, reject an outdated answer, and later display a valid one. |
| Automated software tests | 91 passed | The tested clock, timing, input, and answer-validity rules behave as specified. |

These are results from short, scripted tests. “12 of 12” is not a claim of perfect reliability in ordinary conversations.

## Is TimesFM worth the extra work?

**We do not know yet.**

The clock-only version already performs the basic timekeeping job. It does not need a neural model to count seconds, notice that a deadline passed, or record when information arrived.

In the final test, TimesFM took about **67 milliseconds** to produce its forecast. A simple predictor took about **4 milliseconds**. TimesFM suggested checking again after **1.1 seconds**; the simple predictor suggested **1.3 seconds**. We used TimesFM's suggestion, but this one comparison does not tell us which was better.

The next research question is whether prediction helps the assistant catch changes sooner or make fewer unnecessary model calls. That needs a comparison on the same streams, with the same limits on missed events and response delay. We have not completed that comparison.

For now, the cheapest demonstrated approach is the clock-and-event program. TimesFM is a working optional addition whose benefit remains unproven.

## What went wrong along the way

The first live attempt handled the clock and incoming events correctly, but the language model gave wrong answers about expired deadlines and old measurements. It struggled with the compact technical fields we supplied.

We made those fields explicit: how long the run had been active, how old the last measurement was, and whether a deadline was pending or expired. We also clarified an ambiguous question about whether a deadline was “due now.” The later runs passed after those changes.

We kept the failed runs. They show that simply attaching timestamps to a model is not enough. How the information is presented matters, and the software must enforce timing rules rather than trusting every generated answer.

The earlier static forecasting study was also a detour: it tested questions about numerical sequences, which did not answer the request for awareness of time as it passes. Its results are retained separately.

## What this does—and does not—mean

The combined system now has a practical way to keep time, receive changes while the model is busy, and check whether an answer is still usable.

The language model itself is unchanged. It receives current time information when called; it does not continuously think or maintain an independent internal clock between calls. We have not established a new scientific breakthrough, broad temporal intelligence, or an advantage over all existing approaches. The [research review](realtime-research.md) documents related work.

There are also practical limits. For file input, the program measures age from when it reads a record. It cannot tell how long that record waited before being read, so the original measurement's age may be unknown. Relative deadlines start when read. Replacing or shortening the input file can cause records to be missed. We have not established reliable operation through machine sleep, production restarts, or arbitrary long-running conversations.

## Read the evidence

The tests used a local Qwen3-4B language model in a compressed 4-bit format and Rachitt's Apple Silicon implementation of TimesFM 3. Neither model was retrained. The TimesFM model's non-commercial license applies.

The records below include the exact prompts, clock readings, arriving events, model responses, and decisions to display or reject an answer. Each run identifies the code and model files it used.

- [Final clock-only trial](../evidence/live-clock-20261102-d/summary.json)
- [Final trial with TimesFM](../evidence/live-fused-20261102-b/summary.json)
- [Real file-watcher test output](../evidence/live-watch-20261102-b/test-output.log) and [displayed output](../evidence/live-watch-20261102-b/stdout.log)
- [First failed live trial](../evidence/live-clock-20261102-a/summary.json) and [second attempt](../evidence/live-clock-20261102-b/summary.json)
- [Automated test results](../evidence/verification/pytest.txt)
- [Independent review](realtime-final-review.md)
- [Earlier static-study failure analysis](failure-analysis-v1.md)

<details>
<summary>Additional timing details</summary>

In the final TimesFM trial, the two deadlines were handled 4.33 milliseconds and 0.45 milliseconds after their scheduled times. Ninety-five percent of the recorded gaps between checks were 12.82 milliseconds or shorter. These are measurements from one run, not guarantees about the maximum possible delay.

The TimesFM forecast took 67.39 milliseconds, including the first execution of that input shape in that process. The simple predictor took 3.71 milliseconds. The forecast used 128 regularly spaced values constructed from the measurements received during the live run. The program refuses to fill a long gap in missing data merely to make that input look regular.

Displayed answers are labeled with the time of the information they used. The current clock is reported separately. That distinction matters: showing a new timestamp beside an old answer does not make the old answer current.

</details>
