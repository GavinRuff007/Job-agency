import { Suspense } from "react";
import Loading from "/components/quiz/loading";

async function timeDelay() {
    const delay = 1 + Math.floor(0.15 * 5);
    console.log(`Delay: ${delay}`);

    await timeout(delay * 1000);
}

function timeout(delay) {
    return new Promise((time) => setTimeout(time, delay));
}
    

export default function Answers({
    answers,
    onAnswerSelected,
    selectedAnswerIndex,
}) {
    return answers.map((answer, index) => (
        <li
            key={index}
            onClick={() => onAnswerSelected(answer, index)}
            className="hover:bg-gray-600 cursor-pointer bg-gray-100 dark:bg-gray-900 mx-6 h-16 rounded flex items-center mb-2"
            style={
                selectedAnswerIndex === index
                    ? { backgroundColor: "lightblue" }
                    : {}
            }
        >
            <Suspense fallback={<Loading count={1} />}>
                <span className=" text-grey-700 dark:text-gray-400 text-lg mr-4">
                {timeDelay().then(() => answer)}   
                </span>
            </Suspense>
        </li>
    ));
}
