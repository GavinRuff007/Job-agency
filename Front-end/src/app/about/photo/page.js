import Frame from "@/src/components/modal/Frame";

export default function PhotoPage() {
    const photo =
        "https://platformboy.com/wp-content/uploads/2022/10/%D8%B9%DA%A9%D8%B3-%D9%BE%D8%B1%D9%88%D9%81%D8%A7%DB%8C%D9%84-%D9%BE%D8%B3%D8%B1-%D8%B3%D8%A8%D8%B2%D9%87-%D9%85%D9%88-%D9%85%D8%B4%DA%A9%DB%8C.png";

    return (
        <div className="container mx-auto my-10">
            <div className="w-1/2 mx-auto border border-gray-700">
                <Frame photo={photo} />
            </div>
        </div>
    );
}
