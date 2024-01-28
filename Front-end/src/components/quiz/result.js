// export default function QuizResult({ result, questions }) {
//     // if (result.score/10<5)
//     return (
//         <div className="container p-5">
//             <div className="text-gray-400">
//                 <h3 className="mt-2">نتایج</h3>
//                 <h3 className="mt-2">
//                     به طور کلی {(result.score / 10)} سوالات جواب داده شده
//                 </h3>
                
//                 <p className="mt-2">کل سوالات : {questions.length}</p>
//                 <p className="mt-2">کل امتیاز : {result.score}</p>
//                 <p className="mt-2">سوالات درست : {result.correctAnswers}</p>
//                 <p className="mt-2">سوالات غلط : {result.wrongAnswers}</p>

//                 <button
//                     type="button"
//                     className="my-5 px-6 py-2 text-sm rounded shadow bg-gray-600 hover:bg-gray-400 text-gray-200 w-full cursor-pointer"
//                     onClick={() => window.location.reload()}
//                 >
//                     شروع مجدد آزمون
//                 </button>
//             </div>
//         </div>
//     );
// }

export default function QuizResult({ result, questions }) {
    const items = [
      "agile",
      "Teamwork",
      "Computer Science knowledge",
      "Speed skill",
      "English",
      "attempt",
      "idea",
      "cleanCode",
      "document",
      "polite",
    ];
  
    const renderResults = () => {
      const score = result.score;
      const itemsScores = items.map((item) => {
        const baseScore = Math.floor(Math.random() * 10);
        const bonus = Math.floor((100 - score ) / 10);
        return baseScore + bonus;
      });
  
      return (
        <div className="container p-5">
          <div className="text-gray-400">
            <h3 className="mt-2">نتایج</h3>
            <p className="mt-2">کل سوالات : {questions.length}</p>
            <p className="mt-2">کل امتیاز : {score}</p>
            <p className="mt-2">سوالات درست : {result.correctAnswers}</p>
            <p className="mt-2">سوالات غلط : {result.wrongAnswers}</p>
            {items.map((item, index) => (
              <p key={index} className="mt-2">{item} : {itemsScores[index]}</p>
            ))}
            <button
              type="button"
              className="my-5 px-6 py-2 text-sm rounded shadow bg-gray-600 hover:bg-gray-400 text-gray-200 w-full cursor-pointer"
              onClick={() => window.location.reload()}
            >
              شروع مجدد آزمون
            </button>
          </div>
        </div>
      );
    };
  
    return renderResults();
  }