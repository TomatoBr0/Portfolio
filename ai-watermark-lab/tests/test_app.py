import asyncio

from textual.widgets import TabbedContent

from lab.app import WatermarkLab


def test_tutorial_can_switch_to_lab_without_focus_returning_to_old_tab():
    async def exercise():
        app = WatermarkLab()
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            app.action_learn()
            await pilot.pause()
            await pilot.click("#next")
            await pilot.pause(0.3)
            assert app.lesson == 1
            app.open_lab()
            await pilot.pause()
            assert app.query_one("#tabs", TabbedContent).active == "experiment"
            app.action_learn()
            await pilot.pause()
            assert app.query_one("#tabs", TabbedContent).active == "learn"
    asyncio.run(exercise())
