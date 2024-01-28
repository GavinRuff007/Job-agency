// export default function About() {
//     return (
//         <main>
//             <div>
//                 <h1>توسعه دهنده اپلیکیشن</h1>
//                 <h2>سید محمدامین موسوی نسب 🤗</h2>
//             </div>
//         </main>
//     );
// }
// export default async function About() {
//     await new Promise((resolve) => setTimeout(resolve, 1000));

//     return (
//         <main>
//             <div className="quiz-container" style={{ backgroundColor: "teal" }}>
//                 <h2>بخش اصلی</h2>
//                 {/* <Info /> */}
//             </div>
//         </main>
//     );
// }    
import Image from "next/image";
import Link from "next/link";

import { getServerSession } from "next-auth/next";
// import { redirect } from "next/navigation";
import { options } from "../api/auth/[...nextauth]/options";

export default async function About() {
    const session = await getServerSession(options);

    if (!session) {
        // redirect("/api/auth/signIn?callbackUrl=/about");
    }
    const photo =
    "https://platformboy.com/wp-content/uploads/2022/10/%D8%B9%DA%A9%D8%B3-%D9%BE%D8%B1%D9%88%D9%81%D8%A7%DB%8C%D9%84-%D9%BE%D8%B3%D8%B1-%D8%B3%D8%A8%D8%B2%D9%87-%D9%85%D9%88-%D9%85%D8%B4%DA%A9%DB%8C.png";

        return (
            <main className="p-5 mt-2 bg-gray-50 dark:bg-gray-800 shadow-lg dark:shadow-dark rounded mx-auto w-6/12 ">
                <div className="d-flex justify-content-center align-items-center">
                    <div className="text-gray-300  text-center mb-5">
                        <h1 className="text-lg">سیدمحمدامین موسوی نسب</h1>
                        <h2>برنامه نویس و دانشجوی علم و صنعت</h2>
                    </div>
                    <Link href={`/about/photo`} >
                        <Image
                            alt="عکس پروفایل"
                            src={photo}
                            height={400}
                            width={400}
                            className="mx-auto rounded object-cover aspect-square"
                        />
                    </Link>
                </div>
            </main>
        );
    }

