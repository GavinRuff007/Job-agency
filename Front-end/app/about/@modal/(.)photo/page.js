import Frame from "../../components/frame";
import Modal from "../../components/modal";

export default function PhotoModal() {
    const photo =
        "https://ghorbany.dev/static/media/avatar.ec0231db6078aebd81c7.jpg";

    return (
        <Modal>
            <Frame photo={photo} />
        </Modal>
    );
}
