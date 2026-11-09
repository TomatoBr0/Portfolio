import asyncio

import pytest
from textual.widgets import ContentSwitcher

from lab.app import PAGES, WatermarkLab


@pytest.mark.parametrize('size', [(80, 24), (120, 45)])
def test_next_back_pages_fit_terminal(size):
    async def exercise():
        app = WatermarkLab()
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            assert app.query_one('#pages', ContentSwitcher).current == 'setup'
            for page in PAGES[1:]:
                assert await pilot.click('#next')
                await pilot.pause(0.3)
                assert app.query_one('#pages', ContentSwitcher).current == page
                assert app.query_one('#page-data').region.bottom <= size[1] - 4
                guide = app.query_one('#guide')
                assert guide.region.height > 0
            assert await pilot.click('#back')
            await pilot.pause(0.3)
            assert app.page == len(PAGES) - 2
    asyncio.run(exercise())
