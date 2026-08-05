(() => {
    const transcriptSelector = "#coaches-meeting-transcript";
    const feedSelector = ".headset-feed";
    let observedTranscript = null;
    let transcriptObserver = null;
    let followFrame = null;

    const fitHeadsetToViewport = () => {
        const transcript = document.querySelector(transcriptSelector);
        const timeline = transcript?.querySelector(".headset-timeline");
        const header = timeline?.querySelector(".headset-timeline__header");
        const feed = timeline?.querySelector(feedSelector);
        if (!transcript || !timeline || !header || !feed) return;

        const viewportGutter = 16;
        const transcriptTop = Math.max(transcript.getBoundingClientRect().top, viewportGutter);
        const availableHeight = Math.max(120, window.innerHeight - transcriptTop - viewportGutter);
        const feedHeight = Math.max(80, availableHeight - header.offsetHeight - 6);
        timeline.style.maxHeight = `${availableHeight}px`;
        feed.style.maxHeight = `${feedHeight}px`;
    };

    const followLatestMessage = () => {
        fitHeadsetToViewport();
        const transcript = document.querySelector(transcriptSelector);
        const feed = transcript?.querySelector(feedSelector);
        if (!feed) return;

        feed.scrollTop = feed.scrollHeight;
    };

    const scheduleFollow = () => {
        if (followFrame !== null) window.cancelAnimationFrame(followFrame);
        followFrame = window.requestAnimationFrame(() => {
            followFrame = null;
            followLatestMessage();
        });
    };

    const watchTranscript = () => {
        const transcript = document.querySelector(transcriptSelector);
        if (!transcript || transcript === observedTranscript) return;

        transcriptObserver?.disconnect();
        observedTranscript = transcript;
        transcript.setAttribute("tabindex", "-1");
        transcriptObserver = new MutationObserver(scheduleFollow);
        transcriptObserver.observe(transcript, {
            childList: true, subtree: true, characterData: true,
        });
        scheduleFollow();
    };

    const jumpToHeadset = () => {
        watchTranscript();
        const transcript = document.querySelector(transcriptSelector);
        if (!transcript) return;

        transcript.scrollIntoView({
            behavior: "auto", block: "start",
        });
        transcript.focus({preventScroll: true});
        scheduleFollow();
    };

    document.addEventListener("click", (event) => {
        if (!(event.target instanceof Element)) return;
        if (!event.target.closest("#send-play-call")) return;

        window.setTimeout(jumpToHeadset, 0);
    });

    const pageObserver = new MutationObserver(watchTranscript);
    pageObserver.observe(document.body, {childList: true, subtree: true});
    window.addEventListener("resize", scheduleFollow, {passive: true});
    watchTranscript();
})();
