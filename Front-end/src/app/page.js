// import Link from "next/link";
// // import { getLangs } from "./langs";

// export default async function Home({ params }) {
//     const dict = await getLangs(params.lang);
//     return (
//         <main className="container mx-auto">
//             <div className="text-center">
//                 <h1 className="text-3xl mt-10 mb-10 text-center">
//                     {dict["main"].appName}
//                 </h1>
//                 <Link href="/quiz">
//                     <button
//                         type="button"
//                         className="px-6 py-2 text-sm rounded shadow bg-rose-100 hover:bg-rose-200 text-rose-500 w-72"
//                     >
//                         {dict["main"].appStartBtn}
//                     </button>
//                 </Link>
//             </div>
//         </main>
//     );
// }
import Link from "next/link";

export default function Home() {
    return (
        <main className="container mx-auto">
            <div className="text-center">
                <h1 className="text-3xl mt-10 mb-10 text-center">
                    اپلیکیشن Quiz
                </h1>
                <Link href="/quiz">
                    <button
                        type="button"
                        className="px-6 py-2 text-sm rounded shadow bg-rose-100 hover:bg-rose-200 text-rose-500 w-72"
                    >
                        شروع آزمون
                    </button>
                </Link>
            </div>
        </main>
    );
}
